--[[
Evaluates a trained model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
require 'util.HLogSoftMax'

HSMClass = require 'util.HSMClass'
BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-savefile', 'sample_results.t7', 'save results to')
cmd:option('-model', 'en-large-word-model.t7', 'model checkpoint file')
-- GPU/CPU these params must be passed in because it affects the constructors
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1 = yes, 0 = no)')
cmd:option('-length', 100, 'length of sampled text')
cmd:option('-start_text', '', 'the start of text')
cmd:option('-verbose', 0, 'the char num of start text')
cmd:option('-sample', 1, 'sample or not')
cmd:option('-temperature', 1, 'sample temperature')

cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end

if opt2.cudnn == 1 then
    assert(opt2.gpuid >= 0, 'GPU must be used if using cudnn')
    print('using cudnn')
    require 'cudnn'
end

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
protos = checkpoint.protos
print('opt: ')
print(opt)
print('val_losses: ')
print(checkpoint.val_losses)
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

-- recreate the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
	    .. ', Max word length (incl. padding): ', loader.max_word_l)

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(2, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)
if opt.hsm > 0 then
    hsm_params, hsm_grad_params = model_utils.combine_all_parameters(protos.criterion)
    print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
else
    print('number of parameters in the model: ' .. params:nElement())
end

tokens = opt.tokens

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 then 
        table.insert(u, x_char[{{1,2},t}])
    end
    if opt.use_words == 1 then 
        table.insert(u, x[{{1,2},t}]) 
    end
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

--[[
Sample from the language model

Inputs:
--]]

function sample()
    local T = opt2.length
    local start_text = opt2.start_text
    local verbose = opt2.verbose
    local sample = opt2.sample
    local temperature = opt2.temperature
    local sampled = torch.LongTensor(1, opt2.length)

    if opt.hsm > 0 then
        protos.criterion:change_bias()
    end
    local loss = 0
    -- local token_count = torch.zeros(#idx2word)
    -- local token_loss = torch.zeros(#idx2word)
    local rnn_state = {[0] = init_state}   
    local x = torch.zeros(T)
    local x_char = torch.zeros(T, loader.max_word_l) 
    -- local x, y, x_char = loader:next_batch(split_idx)
    local scores, first_t
    first_t = 1
    if #start_text > 0 then
        if verbose > 0 then
            print('Seeding with: "' .. start_text .. '"')
        end
        -- local x = self:encode_string(start_text):view(1, -1)
        -- local T0 = x:size(2)
        -- sampled[{{}, {1, T0}}]:copy(x)
        -- scores = self:forward(x)[{{}, {T0, T0}}]
        -- first_t = T0 + 1
    else
        if verbose > 0 then
            print('Seeding with uniform probabilities')
        end
        scores = torch.ones(1, #idx2word)
        first_t = 1
    end

    if opt.gpuid >= 0 then
        x = x:float():cuda()
        x_char = x_char:float():cuda()
    end
    protos.rnn:evaluate() 

    local _, next_word = nil, nil
    for t = first_t, T do
        if sample == 0 then
            _, next_word = scores:max(3)
            next_word = next_word[{{}, {}, 1}]
        else
            local probs = torch.div(scores, temperature):double():exp():squeeze()
            probs:div(torch.sum(probs))
            next_word = torch.multinomial(probs, 1):view(1, 1)
        end
        sampled[{{}, {t, t}}]:copy(next_word)

        function split_word(wordind)
            local chars = {char2idx[tokens.START]}
            word = idx2word[wordind[1][1]]

	    local l = utf8.len(word)
            local _, char = nil, nil
            for _, char in utf8.next, word do
                char = utf8.char(char)
                chars[#chars + 1] = char2idx[char]
            end
            chars[#chars + 1] = char2idx[tokens.END]
            return chars
        end

        x[t]=next_word
        chars = split_word(next_word)
	print(loader.max_word_l)
        for i = 1, math.min(#chars, loader.max_word_l) do
            x_char[t][i] = chars[i]
        end
    	local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
    	rnn_state[0] = {}
    	for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
    	prediction = lst[#lst] 
        for i = 1, #idx2word do
            scores[i] = protos.criterion:forward(prediction, i)
        end
    end
    function decode(encoded)
        assert(torch.isTensor(encoded) and encoded:dim() == 1)
        local s = ''
        for i = 1, encoded:size(1) do
            local ind = encoded[i]
            local token = idx2word[ind]
            s = s .. token
        end
        return s
    end
    return decode(sampled[1])
end

test_results = sample()
torch.save(opt2.savefile, test_results)
collectgarbage()

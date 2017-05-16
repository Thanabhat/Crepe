--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University

Modified by Thanabhat Koomsubha
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุูเแโใไๅๆ็่้๊๋์๐๑๒๓๔๕๖๗๘๙"

-- Training data
config.train_data = {}
config.train_data.file = paths.concat(paths.cwd(), "../data/train.t7b")
config.train_data.alphabet = alphabet
config.train_data.min_length = 402
config.train_data.max_length = 2000
config.train_data.batch_size = 32

-- Validation data
config.val_data = {}
config.val_data.file =  paths.concat(paths.cwd(), "../data/valid.t7b")
config.val_data.alphabet = alphabet
config.val_data.min_length = 402
config.val_data.max_length = 2000
config.val_data.batch_size = 32

-- Test data
config.test_data = {}
config.test_data.file =  paths.concat(paths.cwd(), "../data/test.t7b")
config.test_data.alphabet = alphabet
config.test_data.min_length = 402
config.test_data.max_length = 2000
config.test_data.batch_size = 32

-- The model (Small with K Max Pooling at last pooling layer)
local topK = 34
config.model = {}
-- #alphabet x s
config.model[1] = {module = "nn.TemporalConvolution", inputFrameSize = #alphabet, outputFrameSize = 256, kW = 7}
config.model[2] = {module = "nn.ReLU"}
config.model[3] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- K x 256
config.model[4] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 7}
config.model[5] = {module = "nn.ReLU"}
config.model[6] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- K x 256
config.model[7] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[8] = {module = "nn.ReLU"}
-- K x 256
config.model[9] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[10] = {module = "nn.ReLU"}
-- K x 256
config.model[11] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[12] = {module = "nn.ReLU"}
-- K x 256
config.model[13] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[14] = {module = "nn.ReLU"}
config.model[15] = {module = "nn.TemporalDynamicKMaxPooling", minK = topK }
-- 34 x 256
config.model[16] = {module = "nn.Reshape", size = 8704}
-- 8704
config.model[17] = {module = "nn.Linear", inputSize = 8704, outputSize = 1024}
config.model[18] = {module = "nn.ReLU"}
config.model[19] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[20] = {module = "nn.Linear", inputSize = 1024, outputSize = 1024}
config.model[21] = {module = "nn.ReLU"}
config.model[22] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[23] = {module = "nn.Linear", inputSize = 1024, outputSize = 5}
config.model[24] = {module = "nn.LogSoftMax"}


-- The loss
config.loss = nn.ClassNLLCriterion

-- The trainer
config.train = {}
local baseRate = 5e-3
config.train.rates = {[1] = baseRate/1,[30001] = baseRate/2,[60001] = baseRate/4,[90001] = baseRate/8,[120001] = baseRate/16,[150001] = baseRate/32,[180001]= baseRate/64,[210001] = baseRate/128,[240001] = baseRate/256,[270001] = baseRate/512,[300001] = baseRate/1024}
config.train.momentum = 0.9
config.train.decay = 1e-5

-- The tester
config.test = {}
config.test.confusion = true

-- UI settings
config.mui = {}
config.mui.width = 1200
config.mui.scale = 4
config.mui.n = 16

-- Main program
config.main = {}
config.main.type = "torch.CudaTensor"
config.main.eras = 50
config.main.epoches = 10000
config.main.randomize = 5e-2
config.main.dropout = true
config.main.save = paths.concat(paths.cwd())
config.main.details = true
config.main.device = 1
config.main.collectgarbage = 100
config.main.logtime = 5
config.main.debug = false
config.main.test = true

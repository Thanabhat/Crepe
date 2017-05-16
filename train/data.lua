--[[
Data Program for Crepe
By Xiang Zhang @ New York University
--]]

require("image")
local ffi = require("ffi")

-- The class
local Data = torch.class("Data")

function Data:__init(config)
   -- Alphabet settings
   self.alphabet = config.alphabet or "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุูเแโใไๅๆ็่้๊๋์๐๑๒๓๔๕๖๗๘๙"
   self.dict = {}
   local c_ind = 1
   for c in self.alphabet:gmatch(".[\128-\191]*") do
      self.dict[c] = c_ind
      c_ind = c_ind + 1
   end

   self.length = config.length or 1014
   self.min_length = config.min_length
   self.max_length = config.max_length
   self.batch_size = config.batch_size or 128
   self.file = config.file
   self.prob = config.prob
   self.thes = config.thes
   self.padding = config.padding
   self.scale = config.scale
   self.extra = config.extra

   self.config = config
   self.data = torch.load(self.file)

   if self.prob then
      for i = 1, #self.prob - 1 do
	 self.prob[i + 1] = self.prob[i] + self.prob[i + 1]
      end
   end

   if self.thes then
      local data = torch.load(self.thes.file)
      self.thes.data = {}
      for key, defs in pairs(data) do
	 self.thes.data[key] = {}
	 for i, def in ipairs(defs) do
	    for j, word in ipairs(def) do
	       self.thes.data[key][#self.thes.data[key] + 1] = word
	    end
	 end
      end
   end
end

function Data:nClasses()
   return self.data.nClasses[1]
end

function Data:nRow()
   return self.data.index:size(1)
end

function Data:getBatch(inputs, labels, data, extra)
   local data = data or self.data
   local extra = extra or self.extra
   local labels = torch.Tensor(self.batch_size)
   
   local data_length = data.index:size(1)
   local max_batch_index = torch.ceil(data_length / self.batch_size)
   local batch_index = torch.random(max_batch_index)
   
   local current_index = (batch_index - 1) * self.batch_size
   local count = 0
   local max_length = 0
   local s_list = {}
   local s_list_char = {}
   while (count < self.batch_size)
   do
      count = count + 1
      current_index = current_index + 1
      if current_index > data_length then current_index = (batch_index - 1) * self.batch_size + 1 end
      labels[count] = data.content_class[current_index]
      s_list[count] = ffi.string(torch.data(data.content:narrow(1, data.index[current_index], 1)))
      s_list_char[count] = {}
      local ind = 1
      for c in s_list[count]:gmatch(".[\128-\191]*") do
         s_list_char[count][ind] = c
         ind = ind + 1
      end
      if (#s_list_char[count] > max_length) then max_length = #s_list_char[count] end
   end
   if (self.min_length > max_length) then max_length = self.min_length end
   if (self.max_length < max_length) then max_length = self.max_length end

   local inputs = torch.Tensor(self.batch_size, #self.alphabet, max_length)
   for i = 1, self.batch_size do
      self:charArrToTensor(s_list_char[i], max_length, inputs:select(1, i))
   end

   return inputs, labels
end

function Data:iterator(static, data)
   local j = 0
   local done = false
   local data = data or self.data
   local static
   if static == nil then static = true end
   
   return function()
      if done then return end

      local labels = torch.Tensor(self.batch_size)

      local s_list = {}
      local s_list_char = {}
      local max_length = 0
      
      local n = 0
      for k = 1, self.batch_size do
         j = j + 1
         if j > data.index:size(1) then
            j = data.index:size(1)
            done = true
         end
         n = n + 1
         s_list[k] = ffi.string(torch.data(data.content:narrow(1, data.index[j], 1)))
         s_list_char[k] = {}
         local ind = 1
         for c in s_list[k]:gmatch(".[\128-\191]*") do
            s_list_char[k][ind] = c
            ind = ind + 1
         end
         labels[k] = data.content_class[j]
         if (#s_list_char[k] > max_length) then max_length = #s_list_char[k] end
      end
      if (self.min_length > max_length) then max_length = self.min_length end
      if (self.max_length < max_length) then max_length = self.max_length end

      local inputs = torch.Tensor(self.batch_size, #self.alphabet, max_length)
   
      for k = 1, n do
         local data = self:charArrToTensor(s_list_char[k], max_length, inputs:select(1, k))
      end

      return inputs, labels, n
   end
end

function Data:charArrToTensor(charArr, l, input, p)
   local s = {}
   for i = 1, #charArr do
      s[i] = charArr[i]:lower()
   end
   local l = l or #s
   local t = input or torch.Tensor(#self.alphabet, l)
   t:zero()
   for i = #s, math.max(#s - l + 1, 1), -1 do
      if self.dict[s[i]] then
         t[self.dict[s[i]]][#s - i + 1] = 1
      end
   end
end

return Data

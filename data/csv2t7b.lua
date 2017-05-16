--[[
Dataset converter from csv to t7b
By Xiang Zhang @ New York University
--]]

require("io")
require("os")
require("math")
require("paths")
require("torch")
ffi = require("ffi")

-- Configuration table
config = {}
config.input = "train.csv"
config.output = "train.t7b"

-- Parse arguments
cmd = torch.CmdLine()
cmd:option("-input", config.input, "Input csv file")
cmd:option("-output", config.output, "Output t7b file")
params = cmd:parse(arg)
config.input = params.input
config.output = params.output

-- Check file existence
if not paths.filep(config.input) then
   error("Input file "..config.input.." does not exist.")
end
if paths.filep(config.output) then
   io.write("Output file "..config.output.." already exists. Do you want to continue? (y/[n]) ")
   local response = io.read("*line")
   if response:gsub("^%s*(.-)%s*$", "%1") ~= "y" then
      print("Exited because output file "..config.output.." already exists.")
      os.exit()
   end
end

-- Parser function for csv
-- Reference: http://lua-users.org/wiki/LuaCsv
function ParseCSVLine (line,sep) 
   local res = {}
   local pos = 1
   sep = sep or ','
   while true do 
      local c = string.sub(line,pos,pos)
      if (c == "") then break end
      if (c == '"') then
	 -- quoted value (ignore separator within)
	 local txt = ""
	 repeat
	    local startp,endp = string.find(line,'^%b""',pos)
	    txt = txt..string.sub(line,startp+1,endp-1)
	    pos = endp + 1
	    c = string.sub(line,pos,pos) 
	    if (c == '"') then txt = txt..'"' end 
	    -- check first char AFTER quoted string, if it is another
	    -- quoted string without separator, then append it
	    -- this is the way to "escape" the quote char in a quote. example:
	    --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
	 until (c ~= '"')
	 table.insert(res,txt)
	 assert(c == sep or c == "")
	 pos = pos + 1
      else
	 -- no quotes used, just look for the first separator
	 local startp,endp = string.find(line,sep,pos)
	 if (startp) then 
	    table.insert(res,string.sub(line,pos,startp-1))
	    pos = endp + 1
	 else
	    -- no separator found -> use rest of string and terminate
	    table.insert(res,string.sub(line,pos))
	    break
	 end 
      end
   end
   return res
end

print("--- PASS 1: Checking file format and counting samples ---")
count = {}
n = 0
bytecount = 0
fd = io.open(config.input)
for line in fd:lines() do
   n = n + 1
   local content = ParseCSVLine(line)
   nitems = nitems or #content
   if nitems < 2 then
      error("Number of items smaller than 2. Where is the content?")
   end
   if nitems ~= #content then
      error("Inconsistent number of items at line "..n)
   end

   local class = tonumber(content[1])
   if not class then
      error("First item is not a number at line "..n)
   end
   if class <= 0 then
      error("Class index smaller than 1 at line "..n)
   end

   count[class] = count[class] or 0
   count[class] = count[class] + 1

   for i = 2, #content do
      content[i] = content[i]:gsub("\\n", "\n"):gsub("^%s*(.-)%s*$", "%1")
      bytecount = bytecount + content[i]:len() + 1
   end

   if math.fmod(n, 10000) == 0 then
      io.write("\rProcessing line "..n)
      io.flush()
      collectgarbage()
   end
end
fd:close()
collectgarbage()
print("\rNumber of lines processed: "..n)
max_class = 1
for key, val in pairs(count) do
   if key > max_class then
      max_class = key
   end
end
print("Number of classes: "..max_class)
for class = 1, max_class do
   if count[class] == nil or count[class] == 0 then
      error("The largest class index is "..max_class.." but class "..class.." has no samples.")
   end
   print("Number of samples in class "..class..": "..count[class])
end
print("Number of bytes needed to store content: "..bytecount)

print("\n--- PASS 2: Prepare sorting index ---")
count_all = 0
for class = 1, max_class do
   count_all = count_all + count[class]
end

raw_content_length = {}
sorted_index = {}
n = 0
fd = io.open(config.input)
for line in fd:lines() do
   n = n + 1
   local content = ParseCSVLine(line)
   local class = tonumber(content[1])
   local content_all = ""
   for i = 2, #content do
      content[i] = content[i]:gsub("\\n", "\n"):gsub("^%s*(.-)%s*$", "%1")
      if i == 2 then
         content_all = content[i]
      else
         content_all = content_all.." "..content[i]
      end
   end
   raw_content_length[n] = content_all:len()
   sorted_index[n] = n
   if math.fmod(n, 10000) == 0 then
      io.write("\rProcessing line "..n)
      io.flush()
      collectgarbage()
   end
end
fd:close()
collectgarbage()

function compare(a,b)
  return raw_content_length[a] < raw_content_length[b]
end
table.sort(sorted_index, compare)

raw_index_to_sorted_index = {}
for i=1, count_all do
  raw_index_to_sorted_index[sorted_index[i]] = i
end

sorted_cumulative_byte = {}
sorted_cumulative_byte[1] = 1
for i=2, count_all do
  sorted_cumulative_byte[i] = sorted_cumulative_byte[i - 1] + raw_content_length[sorted_index[i - 1]] + 1
end

print("\n--- PASS 3: Constructing data ---")
data = {index = torch.LongTensor(count_all), content = torch.ByteTensor(bytecount), content_class = torch.LongTensor(count_all), nClasses = torch.LongTensor(1)}
data.nClasses[1] = max_class

n = 0
fd = io.open(config.input)
for line in fd:lines() do
   n = n + 1
   local content = ParseCSVLine(line)
   local class = tonumber(content[1])
   local content_all = ""
   for i = 2, #content do
      content[i] = content[i]:gsub("\\n", "\n"):gsub("^%s*(.-)%s*$", "%1")
      if i == 2 then
        content_all = content[i]
      else
        content_all = content_all.." "..content[i]
      end
   end
   
   local sorted_n = raw_index_to_sorted_index[n]
   data.index[sorted_n] = sorted_cumulative_byte[sorted_n]
   ffi.copy(torch.data(data.content:narrow(1, sorted_cumulative_byte[sorted_n], content_all:len() + 1)), content_all)
   data.content_class[sorted_n] = class

   if math.fmod(n, 10000) == 0 then
      io.write("\rProcessing line "..n)
      io.flush()
      collectgarbage()
   end
end
fd:close()
collectgarbage()
print("\rNumber of lines processed: "..n)

print("Saving to "..config.output)
torch.save(config.output, data)
print("Processing done")

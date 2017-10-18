require 'nn'
require 'nngraph'

local attention = {}
function attention.setup( batch_size, seq_length, rnn_size )
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()()) -- prev_hidden states of decoder
	table.insert(inputs, nn.Identity()()) -- annotations from encoder
	local transpose = nn.Transpose({2, 3})(inputs[1])
  local reshaped1 = nn.Reshape(batch_size*seq_length, rnn_size)(inputs[1]):annotate{name='reshaped1'}
  local Ua = nn.Linear(rnn_size, rnn_size)(reshaped1):annotate{name='Ua'}  --fully connected layer for annotations
  local Wa = nn.Linear(rnn_size, rnn_size)(inputs[2]):annotate{name='Wa'}  --fully connected layer for prev hidden state 
	local input_sums = nn.CAddTable()({Ua, Wa}):annotate{name='EltSum'}  -- elementwise add
	--local narrow = nn.Narrow(2,1,1)(input_sums)
	local activation = nn.Tanh()(input_sums)                   -- nonlinear activation function
	local va = nn.Linear(rnn_size, 1)(activation)              -- 2nd fully connected layer
	local reshaped2 = nn.Reshape( batch_size, seq_length )(va)
  local prob = nn.SoftMax()(reshaped2)                       -- Softmax layer
  local unsqueeze = nn.Unsqueeze(3)(prob) 
  local weighted_sum = nn.MM()({transpose, unsqueeze})       -- annotations multiply with softmax probability
  local reshaped3 = nn.Reshape(batch_size, rnn_size)(weighted_sum)
	table.insert(outputs, reshaped3)
	--table.insert(outputs, weighted_sum)
  return nn.gModule(inputs, outputs)
end
return attention
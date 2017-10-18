require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
--local EQ_ALIGN = require 'misc.EQ_ALIGNMENT'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-dec_model','','path to model to evaluate')
cmd:option('-aspect_net', {'/home/iis/Documents/research/aspect_caption/aspecttalk/eval_model/model_id_comp_161014_140000.t7',
                           '/home/iis/Documents/research/aspect_caption/aspecttalk/eval_model/model_id_color_161014_140000.t7',
                           '/home/iis/Documents/research/aspect_caption/aspecttalk/eval_model/model_id_subject_161014_140000.t7'},
                           'path to a model checkpoint to initialize model weights from, including a shared cnn. Empty = don\'t')
-- Basic options
cmd:option('-batch_size', 0, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', 100, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 1, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
-- Encoder Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')


-- For evaluation on a folder of images:
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
-- For evaluation on MSCOCO images from some split:
cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-index_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dec_model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.dec_model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
  print (v, opt[v])
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
if string.len(opt.image_folder) == 0 then
  loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
else
  print ('Load raw images....')
  loader = DataLoaderRaw{folder_path = opt.image_folder, index_json = opt.index_json}
end

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local vocab = nil
local encoder = {}
encoder.aspect = nn.ConcatTable()

-- load aspect net from file
print('*******************************************************')
assert( #opt.aspect_net > 0, 'aspect net should be loaded...')
for i,model in ipairs(opt.aspect_net) do
  local loaded_checkpoint = torch.load(model)
  if encoder.cnn == nil then
    encoder.cnn = loaded_checkpoint.protos.cnn
    print('Load cnn model....')
  end
  if vocab == nil then
    vocab = loaded_checkpoint.vocab
    print('Load dictionary....')
  end
  loaded_checkpoint.protos.lm:setLMtype('encoder')
	loaded_checkpoint.protos.lm:setModelName(model)
  print (string.format('Load %d encoder: %s', i, model))
  encoder.aspect:add(loaded_checkpoint.protos.lm)
end
encoder.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually

print('*******************************************************')
print('Create decoder netowrk....')
local decoder = {} 
local checkpoint = torch.load(opt.dec_model)
print (string.format('Load decoder from: %s', i, opt.dec_model))
decoder.lm = checkpoint.protos
decoder.crit = nn.LanguageModelCriterion()

if opt.gpuid >= 0 then 
  for k,v in pairs(encoder) do v:cuda() end
  for k,v in pairs(decoder) do v:cuda() end
end

local params, grad_params = decoder.lm:getParameters()
print('total number of parameters in Decoder LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())
-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_images = utils.getopt(evalopt, 'num_images', true)

  encoder.cnn:evaluate()
  encoder.aspect:evaluate()
  decoder.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}

  while true do
    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)
    -- forward the model to get loss
    local feats = encoder.cnn:forward(data.images)
    local expanded_feats = encoder.expander:forward(feats)
    local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local all_states = {}
    encoder.aspect:apply(function(module)
      if torch.typename(module) == 'nn.LanguageModel' then
        local seq, seqLogprobs, states = module:sample(expanded_feats, sample_opts)
        table.insert(all_states, states) 
      end
    end)

		local h_concat = torch.cat({all_states[1], all_states[2], all_states[3]}, 1)
    local h_transpose = h_concat:permute(2,1,3)    

    -- evaluate loss if we have the labels
    local loss = 0
    if data.labels then
      local logprobs = decoder.lm:forward{expanded_feats, data.labels, h_transpose}
      local loss = decoder.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1
    end

    -- forward the generator to also get generated samples for each image 
    local seq, att_result = decoder.lm:sample(expanded_feats, opt, h_transpose)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents,5 do
      local idx = (k-1)/5 + 1
      local entry = {image_id = data.names[idx], caption = sents[k]}
      if opt.dump_path == 1 then
        entry.file_name = data.infos[k].file_path
      end
      table.insert(predictions, entry)
      if opt.dump_images == 1 then
        -- copy the raw image to vis/ folder
        local cmd = 'cp "' .. path.join(opt.image_root, data.infos[idx].file_path) .. '" vis/imgs/img' .. #predictions .. '.jpg' -- bit gross
        print(cmd)
        os.execute(cmd) -- dont think there is cleaner way in Lua
      end
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then
      print(string.format('evaluating performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats, snapshots_prob
end

-- main
local loss, split_predictions, lang_stats, snapshots_prob = eval_split(opt.split, {num_images = opt.num_images})
print('loss: ', loss)
if lang_stats then
  print(lang_stats)
end
if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end

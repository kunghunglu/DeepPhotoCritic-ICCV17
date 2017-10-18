require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','script/prepro_combined/prepro2/combined_captions.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','script/prepro_combined/prepro2/combined_captions.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-aspect_net', {'./model/CriticNet_comp.t7',
                           './model/CriticNet_color.t7',
                           './model/CriticNet_subject.t7'},
                           'path to a model checkpoint to initialize model weights from, including a shared cnn. Empty = don\'t')
cmd:option('-dec_model', './model/fintuneCOCO_dec_10390.t7' )
-- Model settings
cmd:option('-rnn_size',768,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',768,'the encoding size of each token in the vocabulary, and the image.')
-- Encoder Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

-- Optimization: General
cmd:option('-max_iters', 150000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',10,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 25000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 7500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', './checkpoint', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
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
    print('Load CNN model....')
  end
  if vocab == nil then
    vocab = loaded_checkpoint.vocab
    print('Load dictionary....')
  end
  loaded_checkpoint.protos.lm:setLMtype('encoder')
  print (string.format('Load %d encoder: %s', i, model))
  encoder.aspect:add(loaded_checkpoint.protos.lm)
end
encoder.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually

print('*******************************************************')
print('Create Decoder netowrk....')
local lmOpt = {}
lmOpt.vocab_size = loader:getVocabSize()
lmOpt.input_encoding_size = opt.input_encoding_size
lmOpt.rnn_size = opt.rnn_size
lmOpt.num_layers = 1
lmOpt.dropout = opt.drop_prob_lm
lmOpt.seq_length = loader:getSeqLength()
lmOpt.batch_size = opt.batch_size * opt.seq_per_img
lmOpt.context = true
lmOpt.LMtype = 'decoder'
lmOpt.num_encoders = #(opt.aspect_net)
print (string.format('Number of encoders: %d', lmOpt.num_encoders))
local decoder = {}
decoder.lm = nn.LanguageModel(lmOpt)
decoder.crit = nn.LanguageModelCriterion()

if opt.dec_model ~= '' then
  local loaded_decoder = torch.load(opt.dec_model)
  print (string.format('Load decoder: %s', opt.dec_model))
  decoder.lm.core = loaded_decoder.protos.lm.core:clone()
  decoder.lm.lookup_table = loaded_decoder.protos.lm.lookup_table:clone()
end
loaded_decoder = nil
loaded_checkpoint = nil

local lm_modules = decoder.lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end -- add gradient term to model (turn on the gradient) 

if opt.gpuid >= 0 then 
  for k,v in pairs(encoder) do v:cuda() end
  for k,v in pairs(decoder) do v:cuda() end 
end

local params, grad_params = decoder.lm:getParameters()
print('total number of parameters in Decoder LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())


-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_decoder = decoder.lm:clone()
thin_decoder.core:share(decoder.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_decoder.lookup_table:share(decoder.lm.lookup_table, 'weight', 'bias')
thin_decoder.attention:share(decoder.lm.attention, 'weight', 'bias')

decoder.lm:createClones()
collectgarbage() -- "yeah, sure why not

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  encoder.cnn:evaluate()
  encoder.aspect:evaluate()
  decoder.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()

  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the aspect nets to get hidden annotations
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

    -- forward the generator to get loss
    local logprobs = decoder.lm:forward{expanded_feats, data.labels, h_transpose}
    local loss = decoder.crit:forward(logprobs, {data.labels, data.scores})
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the generator to also get generated samples for each image 
    local seq = decoder.lm:sample(expanded_feats, opt, h_transpose)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents,5 do  -- due to the architecture of attention model, we duplicate each image for 5 times
      local idx = (k-1)/5 + 1
      local entry = {image_id = data.infos[idx].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  -- call MSCOCO evaluation, not useful in our work...
  --[[
  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end
  ]]
  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  encoder.cnn:evaluate()
  encoder.aspect:evaluate()
  decoder.lm:training()
  grad_params:zero()
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation

  -- data.images: Nx3x224x224, N: batch size
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img
  -- forward the ConvNet on images (most work happens here)
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
  local h_concat = torch.cat({all_states[1], all_states[2], all_states[3]}, 1) --num_aspect X seq_length X rnn_size
  local h_transpose = h_concat:permute(2,1,3) --seq_length X num_aspect X rnn_size 
  local logprobs = decoder.lm:forward{expanded_feats, data.labels, h_transpose}

  -- forward the language model criterion
  local loss = decoder.crit:forward( logprobs, data.labels )

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = decoder.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = unpack(decoder.lm:backward({expanded_feats, data.labels}, dlogprobs))

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local iter = 0
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

while true do  
  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter > 0 and iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss

    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. '_' .. iter ..'_'.. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write model and vocab to .t7 file
    checkpoint.protos = thin_decoder
    checkpoint.vocab = loader:getVocab()
    torch.save(checkpoint_path .. '.t7', checkpoint)
    print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter >= opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5,  math.floor(frac) )
    learning_rate = learning_rate * decay_factor -- set the decayed rate

    if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
      print ('frac', math.floor(frac) )
      print ('decay_factor', decay_factor )  
      print ('Now learning_rate is ', learning_rate)
    end
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end

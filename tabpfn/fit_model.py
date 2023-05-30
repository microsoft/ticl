import time
from datetime import datetime

import torch

from scripts.model_builder import get_model, save_model

from scripts.model_configs import *

from priors.utils import uniform_int_sampler_f

from notebook_utils import *


# In[3]:


large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'


# In[11]:


device = 'cuda'
base_path = '.'
max_features = 100



def train_function(config_sample, i, add_name='', state_dict=None, load_model_strict=True):
    start_time = time.time()
    N_epochs_to_save = 100
    save_every = max(1, config_sample['epochs'] // N_epochs_to_save)
    
    def save_callback(model, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        # if ((time.time() - start_time) / (maximum_runtime * 60 / N_epochs_to_save)) > model.last_saved_epoch:
        print(f"epoch: {epoch* config_sample['epochs']} save_every: {save_every}")
        if (epoch * config_sample['epochs']) % save_every == 0 or model.last_saved_epoch <10:
            file_name = f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt'
            print(f'Saving model to {file_name}')
            config_sample['epoch_in_training'] = epoch
            save_model(model, base_path, file_name,
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
    
    model = get_model(config_sample
                      , device
                      , should_train=True
                      , verbose=1
                      , epoch_callback = save_callback, state_dict=state_dict, load_model_strict=load_model_strict)
    
    return model



def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = 'fit_vanilla_lr0001_warm_start'
    
    config['epochs'] = 12000
#    config['epochs'] = 1
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string




config, model_string = reload_config(longer=1)

# diff
config['output_multiclass_ordered_p'] = 0.
del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

config['multiclass_type'] = 'rank'
del config['differentiable_hyperparameters']['multiclass_type']

config['sampling'] = 'normal' # vielleicht schlecht?
del config['differentiable_hyperparameters']['sampling']

config['pre_sample_causes'] = True
# end diff

config['multiclass_loss_type'] = 'nono' # 'compatible'
config['normalize_to_ranking'] = False # False

config['categorical_feature_p'] = .2 # diff: .0

# turn this back on in a random search!?
config['nan_prob_no_reason'] = .0
config['nan_prob_unknown_reason'] = .0 # diff: .0
config['set_value_to_nan'] = .1 # diff: 1.

config['normalize_with_sqrt'] = False

config['new_mlp_per_example'] = True
config['prior_mlp_scale_weights_sqrt'] = True
config['batch_size_per_gp_sample'] = None

config['normalize_ignore_label_too'] = False

config['differentiable_hps_as_style'] = False
config['max_eval_pos'] = 1000

config['random_feature_rotation'] = True
config['rotate_normalized_labels'] = True

config["mix_activations"] = False # False heisst eig True

#config['lr'] = 0.00005
config['lr'] = 0.0001
#config['nlayers'] = 18
config['nlayers'] = 12
# config['nlayers'] = 6
config['emsize'] = 512
#config['emsize'] = 1024
# config['nhead'] = config['emsize'] // 128
config['nhead'] = 4
config['bptt'] = 1024+128
config['y_encoder'] = "one_hot"
#config['encoder'] = 'featurewise_mlp'
    
#config['aggregate_k_gradients'] = 8
# config['aggregate_k_gradients'] = 8
config['aggregate_k_gradients'] = 32
# config['batch_size'] = 16 * config['aggregate_k_gradients']  # DEFAULT
config['batch_size'] = 512 // 32
#config['num_steps'] = 1024//config['aggregate_k_gradients']
config['num_steps'] = 1024
config['epochs'] = 300
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config['weight_decay'] = 1e-5

#config['model_maker'] = 'mlp'
config['model_maker'] = False
config['output_attention'] = True
config['special_token'] = False
config['decoder_embed_dim'] = 640
config['decoder_hidden_size'] = 1024
config['decoder_two_hidden_layers'] = False
config['min_eval_pos'] = 2
config['predicted_hidden_layer_size'] = 128
config['no_double_embedding'] = True


config_sample = evaluate_hypers(config)


# ## Training
warm_start_weights = "models_diff/prior_diff_real_checkpointfit_vanilla_lr0003_multiclass_05_26_2023_20_55_24_n_0_epoch_23.cpkt"
# warm_start_weights = None
model_dict = None

if warm_start_weights is not None:
    model_state, optimizer_state, _ = torch.load(
        warm_start_weights, map_location='cpu')
    module_prefix = 'module.'
    model_dict = {k.replace(module_prefix, ''): v for k, v in model_state.items()}


model = train_function(config_sample, i=0, add_name=model_string, state_dict=model_dict, load_model_strict=warm_start_weights is None)

rank = 0
if 'LOCAL_RANK' in os.environ:
    # launched with torch.distributed.launch
    rank = int(os.environ["LOCAL_RANK"])
    print('torch.distributed.launch and my rank is', rank)

if rank == 0:
    i = 0
    save_model(model[2], base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_on_exit.cpkt',
                   config_sample)

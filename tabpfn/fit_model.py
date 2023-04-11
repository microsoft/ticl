#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[2]:


import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
from scripts.differentiable_pfn_evaluation import eval_model_range
from scripts.model_builder import get_model, get_default_spec, save_model, load_model
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow

from scripts.model_configs import *

from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from priors.utils import plot_prior, plot_features
from priors.utils import uniform_int_sampler_f

from scripts.tabular_metrics import calculate_score_per_method, calculate_score
from scripts.tabular_evaluation import evaluate

from priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from scripts import tabular_metrics
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


# In[12]:


def print_models(model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt'))
        print()


# In[13]:


def train_function(config_sample, i, add_name='', state_dict=None):
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
                      , epoch_callback = save_callback, state_dict=state_dict)
    
    return model


# ## Define prior settings

# In[14]:


def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = '_predict_linear_coefficients_nlayer_6'
    
    config['epochs'] = 12000
#    config['epochs'] = 1
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string


# ## Visualize Prior samples

# In[15]:


config, model_string = reload_config(longer=1)

config['bptt_extra_samples'] = None

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
config['nlayers'] = 6
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
config['aggregate_k_gradients'] = 4
# config['batch_size'] = 16 * config['aggregate_k_gradients']  # DEFAULT
config['batch_size'] = 512
#config['num_steps'] = 1024//config['aggregate_k_gradients']
config['num_steps'] = 1024//16//2
config['epochs'] = 300
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config['weight_decay'] = 1e-5

config['model_maker'] = True
config['min_eval_pos'] = 2

config_sample = evaluate_hypers(config)


# In[16]:


# config_sample['batch_size'] = 4
# model = get_model(config_sample, device, should_train=False, verbose=2) # , state_dict=model[2].state_dict()
# (hp_embedding, data, _), targets, single_eval_pos = next(iter(model[3]))

#from utils import normalize_data
#fig = plt.figure(figsize=(8, 8))
#N = 100
#plot_features(data[0:N, 0, 0:4], targets[0:N, 0], fig=fig)

#d = np.concatenate([data[:, 0, :].T, np.expand_dims(targets[:, 0], -1).T])
#d[np.isnan(d)] = 0
#c = np.corrcoef(d)
#plt.matshow(np.abs(c), vmin=0, vmax=1)
#plt.show()


# In[ ]:





# ## Training

# In[19]:


# model = get_model(config_sample, device, should_train=True, verbose=1)

model = train_function(config_sample, i=0, add_name=model_string)
# import pickle
# with open(f"all_{model_string}.pickle", "wb") as f:
#    pickle.dump(model[:3], f, protocol=-1)
rank = 0
if 'LOCAL_RANK' in os.environ:
    # launched with torch.distributed.launch
    rank = int(os.environ["LOCAL_RANK"])
    print('torch.distributed.launch and my rank is', rank)

if rank == 0:
    i = 0
    save_model(model[2], base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_on_exit.cpkt',
                   config_sample)

# In[ ]:

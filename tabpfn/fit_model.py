from datetime import datetime
import os

import torch

from scripts.model_builder import get_model, save_model

from scripts.model_configs import get_prior_config, evaluate_hypers

from priors.utils import uniform_int_sampler_f

# from notebook_utils import *



large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'



device = 'cuda'
base_path = '.'
max_features = 100

def reload_config(config_type='causal'):
    config = get_prior_config(config_type=config_type)
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    config['recompute_attn'] = True
    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    return config


config = reload_config()

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

config['lr'] = 0.0003
#config['lr'] = 0.0001
config['nlayers'] = 18
# config['nlayers'] = 12
# config['nlayers'] = 6
config['emsize'] = 2048
# config['emsize'] = 1024
config['nhead'] = config['emsize'] // 128
# config['nhead'] = 16
# config['nhead'] = 4
config['bptt'] = 1024+128
config['y_encoder'] = "one_hot"
#config['encoder'] = 'featurewise_mlp'
    
#config['aggregate_k_gradients'] = 8
config['aggregate_k_gradients'] = 32
config['batch_size'] = 16
config['num_steps'] = 1024
# config['num_steps'] = 32
config['epochs'] = 300

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config['weight_decay'] = 1e-5

config['model_maker'] = 'mlp'
# config['model_maker'] = False
config['output_attention'] = True
config['special_token'] = False
config['decoder_embed_dim'] = 2048
config['decoder_hidden_size'] = 2048
config['decoder_two_hidden_layers'] = False
config['min_eval_pos'] = 2
config['predicted_hidden_layer_size'] = 128
config['no_double_embedding'] = True

config_sample = evaluate_hypers(config)


# ## Training
# warm_start_weights = "models_diff/prior_diff_real_checkpointfit_vanilla_lr0001_warm_start_debugging_blabla_multiclass_05_31_2023_19_26_33_n_0_epoch_12.cpkt"
warm_start_weights = None

model_string = 'vanilla_lr0001_new'
model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
model_dict = None
if warm_start_weights is not None:
    model_state, optimizer_state, _ = torch.load(
        warm_start_weights, map_location='cpu')
    module_prefix = 'module.'
    model_dict = {k.replace(module_prefix, ''): v for k, v in model_state.items()}

save_every = 10

def save_callback(model, epoch):
    if not hasattr(model, 'last_saved_epoch'):
        model.last_saved_epoch = 0
    log_file = f'log/{model_string}.log'
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} loss {model.losses[-1]} learning_rate {model.learning_rates[-1]}\n')
    if (epoch == "on_exit") or epoch % save_every == 0:
        file_name = f'models_diff/{model_string}_epoch_{epoch}.cpkt'
        with open(log_file, 'a') as f:
            f.write(f'Saving model to {file_name}\n')
        print(f'Saving model to {file_name}')
        config_sample['epoch_in_training'] = epoch
        config_sample['learning_rates'] = model.learning_rates
        config_sample['losses'] = model.losses
        save_model(model, base_path, file_name, config_sample)

model = get_model(config_sample
                    , device
                    , should_train=True
                    , verbose=1
                    , epoch_callback=save_callback, state_dict=model_dict, load_model_strict=warm_start_weights is None)    

rank = 0
if 'LOCAL_RANK' in os.environ:
    # launched with torch.distributed.launch
    rank = int(os.environ["LOCAL_RANK"])
    print('torch.distributed.launch and my rank is', rank)

if rank == 0:
    save_callback(model[1], "on_exit")

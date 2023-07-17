from datetime import datetime
import os
import torch
import mlflow

from scripts.model_builder import get_model, save_model
from scripts.model_configs import get_prior_config, evaluate_hypers

from priors.utils import uniform_int_sampler_f
import argparse

device = 'cuda'
base_path = '.'

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

config['output_attention'] = True
config['special_token'] = False

#config['lr'] = 0.00003
config['lr'] = 0.0001
# config['nlayers'] = 18
config['nlayers'] = 12
# config['emsize'] = 2048
# config['emsize'] = 1024
config['hid_factor'] = 2
config['emsize'] = 512
# config['emsize'] = 512
# config['emsize'] = 256
config['nhead'] = config['emsize'] // 128
# config['nhead'] = 16
# config['nhead'] = 4
config['bptt'] = 1024+128
config['y_encoder'] = "one_hot"
    
# config['aggregate_k_gradients'] = 8
config['aggregate_k_gradients'] = 1
config['batch_size'] = 32
config['num_steps'] = 1024
config['epochs'] = 2000

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config['weight_decay'] = 0

#config['model_maker'] = 'perceiver'
config['model_maker'] = 'mlp'
#config['model_maker'] = False
config['decoder_embed_dim'] = config['emsize'] 
config['decoder_hidden_size'] = config['emsize'] * config['hid_factor'] 
config['decoder_two_hidden_layers'] = False
config['min_eval_pos'] = 2
config['predicted_hidden_layer_size'] = 128

config['no_double_embedding'] = True
config['prenorm'] = True

config_sample = evaluate_hypers(config)



if 'LOCAL_RANK' in os.environ:
    # launched with torch.distributed.launch
    rank = int(os.environ["LOCAL_RANK"])
    print('torch.distributed.launch and my rank is', rank)
    config_sample['num_gpus'] = int(os.environ["WORLD_SIZE"])
else:
    # Single GPU training, get GPU ID from command line
    parser = argparse.ArgumentParser(description='Train Mothernet')
    parser.add_argument('-g', '--gpu-id', nargs=1, type=int, help='GPU id')
    args = parser.parse_args()
    if args.gpu_id is not None:
        device = f'cuda:{args.gpu_id[0]}'
    torch.set_num_threads(24)
    config_sample['num_gpus'] = 1


# ## Training
#warm_start_weights = "models_diff/perceiver_output_128_emsize_512_nlayers_12_06_28_2023_22_09_12_epoch_430.cpkt"
warm_start_weights = None
continue_old_config = False

model_maker_string = "perceiver" if config['model_maker'] == "perceiver" else ('mothernet' if config['model_maker'] == "mlp" else "tabpfn")
model_string = f"{model_maker_string}_{config['predicted_hidden_layer_size']}_decoder_{config['decoder_hidden_size']}_emsize_{config['emsize']}_nlayers_{config['nlayers']}_steps_{config['num_steps']}_bs_{config['batch_size'] * config['aggregate_k_gradients'] * config_sample['num_gpus']}_lr_{config['lr']}_{config_sample['num_gpus']}_gpu{'s' if config_sample['num_gpus'] > 1 else ''}"
# model_string = 'perceiver_output_128_emsize_512_nlayers_12_steps_4096_batch_16_one_gpu'
model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
model_dict = None
if warm_start_weights is not None:
    model_state, optimizer_state, old_config = torch.load(
        warm_start_weights, map_location='cpu')
    module_prefix = 'module.'
    model_dict = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    if continue_old_config:
        config_sample = old_config


save_every = 10

def save_callback(model, epoch):
    if not hasattr(model, 'last_saved_epoch'):
        model.last_saved_epoch = 0
    log_file = f'log/{model_string}.log'
    if epoch == "start":
        print(f"Starting training of model {model_string}")
        return
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} loss {model.losses[-1]} learning_rate {model.learning_rates[-1]}\n')
    if epoch != "on_exit":
        mlflow.log_params({k:v for k, v in config_sample.items() if isinstance(v, (int, float, str)) and k != 'epoch_in_training'})
        mlflow.log_metric(key="wallclock_time", value=model.wallclock_times[-1], step=epoch)
        mlflow.log_metric(key="loss", value=model.losses[-1], step=epoch)
        mlflow.log_metric(key="learning_rate", value=model.learning_rates[-1], step=epoch)
    
    if (epoch == "on_exit") or epoch % save_every == 0:
        file_name = f'models_diff/{model_string}_epoch_{epoch}.cpkt'
        with open(log_file, 'a') as f:
            f.write(f'Saving model to {file_name}\n')
        print(f'Saving model to {file_name}')
        config_sample['epoch_in_training'] = epoch
        config_sample['learning_rates'] = model.learning_rates
        config_sample['losses'] = model.losses
        config_sample['wallclock_times'] = model.wallclock_times

        save_model(model, base_path, file_name, config_sample)

with mlflow.start_run(run_name=model_string):
    mlflow.set_tracking_uri("http://20.114.249.177:5000")
    model = get_model(config_sample
                        , device
                        , should_train=True
                        , verbose=1
                        , epoch_callback=save_callback, state_dict=model_dict, load_model_strict=continue_old_config)    

rank = 0


if rank == 0:
    save_callback(model[1], "on_exit")

from datetime import datetime
import os
import torch
import mlflow
import sys

from tabpfn.scripts.model_builder import get_model, save_model
from tabpfn.scripts.model_configs import get_prior_config, evaluate_hypers
from tabpfn.utils import compare_dicts

from tabpfn.priors.utils import uniform_int_sampler_f
import argparse
import socket
import shutil


def main(argv):
    config = get_prior_config(config_type='causal')
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    config['recompute_attn'] = True
    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False

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

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = False # False heisst eig True

    config['output_attention'] = True
    config['y_encoder'] = "one_hot"
    config['efficient_eval_masking'] = True
    config['min_eval_pos'] = 2


    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        config['num_gpus'] = int(os.environ["WORLD_SIZE"])
        raise ValueError("Gave up on multi-gpu for now")
    else:
        rank = 0

    # Single GPU training, get GPU ID from command line
    parser = argparse.ArgumentParser(description='Train Mothernet')
    parser.add_argument('-g', '--gpu-id', type=int, help='GPU id')
    parser.add_argument('-e', '--em-size', type=int, help='embedding size', default=512)
    parser.add_argument('-n', '--num-steps', type=int, help='number of steps per epoch')
    parser.add_argument('-E', '--epochs', type=int, help='embedding size', default=2000)
    parser.add_argument('-d', '--decoder-em-size', type=int, help='decoder embedding size')
    parser.add_argument('-H', '--decoder-hidden-size', type=int, help='decoder hidden size')
    parser.add_argument('-l', '--learning-rate', type=float, help='maximum learning rate', default=0.00003)
    parser.add_argument('-N', '--num-layers', type=int, help='number of transformer layers', default=12)
    parser.add_argument('-k', '--agg-gradients', type=int, help='number steps to aggregate gradient over', default=1)
    parser.add_argument('-b', '--batch-size', type=int, help='physical batch size', default=8)
    parser.add_argument('-m', '--model-maker', type=str, help='model maker kind. MLP for mothernet, Perceiver or False for TabPFN', default='mlp')
    parser.add_argument('-A', '--no-adaptive-batch-size', help='Wether to progressively increase effective batch size.', action='store_true')
    parser.add_argument('-w', '--weight-decay', type=float, help='Weight decay for AdamW.', default=0)
    parser.add_argument('-f', '--load-file', help='Warm start from this file')
    parser.add_argument('-c', '--continue-run', help='Whether to read the old config when warm starting', action='store_true')
    parser.add_argument('-s', '--load-strict', help='Whether to load the architecture strictly when warm starting', action='store_true')
    parser.add_argument('-r', '--restart-scheduler', help='Whether to restart the scheduler when warm starting', action='store_true')
    parser.add_argument('-D', '--double-embedding', help='whether to reuse transformer embedding for mlp', action='store_true')
    parser.add_argument('-S', '--special-token', help='whether add a special output token in the first layer as opposed to having one in the last attention layer. If True, decoder-em-size is ignored.', action='store_true')
    parser.add_argument('-T', '--decoder-two-hidden-layers', help='whether to use two hidden layers for the decoder', action='store_true')
    parser.add_argument('-C', '--use-cpu', help='whether to use cpu', action='store_true')
    parser.add_argument('-L', '--num-predicted-hidden-layers', type=int, help='number of predicted hidden layers', default=1)
    parser.add_argument('-W', '--weight-embedding-rank', type=int, help='Rank of weights in predicted network. If None, no shared parameters are learned.')
    parser.add_argument('-P', '--predicted-hidden-layer-size', type=int, help='Size of hidden layers in predicted network.', default=128)
    parser.add_argument('-R', '--create-new-run', help="Create as new MLFLow run, even if continuing", action='store_true')
    parser.add_argument('-Q', '--learning-rate-schedule', help="Learning rate schedule. Cosine, constant or exponential", default='cosine')
    parser.add_argument('-U', '--warmup-epochs', type=int, help="Number of epochs to warm up learning rate (linear climb)", default=20)
    parser.add_argument('--experiment', help="Name of mlflow experiment", default='Default')
    parser.add_argument('-B', '--base-path', default='.')
    parser.add_argument('--no-pre-norm', action='store_true')

    args = parser.parse_args(argv)
    if args.gpu_id is not None:
        if args.use_cpu:
            raise ValueError("Can't use cpu and gpu at the same time")
        device = f'cuda:{args.gpu_id}'
    elif args.use_cpu:
        device = 'cpu'

    if args.create_new_run and not args.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = args.base_path

    torch.set_num_threads(24)
    config['num_gpus'] = 1

    config['lr'] = args.learning_rate
    config['nlayers'] = args.num_layers
    config['emsize'] = args.em_size
    config['aggregate_k_gradients'] = args.agg_gradients
    config['batch_size'] = args.batch_size
    config['model_maker'] = args.model_maker
    config['adaptive_batch_size'] = not args.no_adaptive_batch_size
    config['weight_decay'] = args.weight_decay
    config['special_token'] = args.special_token
    config['device'] = device
    config['predicted_hidden_layers'] = args.num_predicted_hidden_layers
    config['learning_rate_schedule'] = args.learning_rate_schedule
    config['warmup_epochs'] = args.warmup_epochs
    config['train_mixed_precision'] = False
    config['pre_norm'] = not args.no_pre_norm

    warm_start_weights = args.load_file
    config['no_double_embedding'] = not args.double_embedding
    config['hid_factor'] = 2
    config['nhead'] = config['emsize'] // 128
        
    config['num_steps'] = args.num_steps or 1024 * 64 // config['batch_size'] // config['aggregate_k_gradients']
    config['epochs'] = args.epochs
    config['weight_embedding_rank'] = args.weight_embedding_rank

    if config['model_maker'] == 'perceiver':
        config['max_eval_pos'] = 8 * 1000
        config['bptt'] = 8 * 1024+128
    else:
        config['max_eval_pos'] = 1000
        config['bptt'] = 1024+128
        
    config['decoder_embed_dim'] = args.decoder_em_size or config['emsize'] 
    config['decoder_hidden_size'] = args.decoder_hidden_size or config['emsize'] * config['hid_factor'] 
    config['decoder_two_hidden_layers'] = args.decoder_two_hidden_layers
    config['predicted_hidden_layer_size'] = args.predicted_hidden_layer_size
    config['warm_start_from'] = warm_start_weights
    config['continue_old_config'] = args.continue_run

    config_sample = evaluate_hypers(config)
        
    model_state, optimizer_state, scheduler = None, None, None
    if warm_start_weights is not None:
        model_state, old_optimizer_state, old_scheduler, old_config = torch.load(
            warm_start_weights, map_location='cpu')
        module_prefix = 'module.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        if args.continue_run:
            config_sample = old_config
            config_sample['device'] = device
            optimizer_state = old_optimizer_state
            if not args.restart_scheduler:
                scheduler = old_scheduler
        else:
            print("WARNING warm starting with new settings")
            compare_dicts(config_sample, old_config, all=True)

    if args.continue_run:
        model_string = warm_start_weights.split("/")[-1].split("_epoch_")[0]
        if args.create_new_run:
            model_string = model_string + '_continue_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    else:
        model_maker_string = "perceiver" if config_sample['model_maker'] == "perceiver" else ('mn' if config_sample['model_maker'] == "mlp" else "tabpfn")

        default_args_dict = vars(parser.parse_args([]))
        args_dict = vars(args)
        config_string = ""
        for arg in parser._actions:
            if arg.option_strings:
                k = arg.dest
                if k in ['run_id', 'load_file', 'use_cpu', 'continue_run', 'restart_scheduler', 'load_strict', 'gpu_id', 'help', 'base_path', 'create_new_run', 'experiment'] or k not in args_dict:
                    continue
                v = args_dict[k]
                short_name = arg.option_strings[0].replace('-', '')
                if v != default_args_dict[k]:
                    config_string += f"_{short_name}{v}"
        gpu_string = f"_{config_sample['num_gpus']}_gpu{'s' if config_sample['num_gpus'] > 1 else ''}" if config_sample['device'] != 'cpu' else '_cpu'
        model_string = f"{model_maker_string}{config_string}{gpu_string}{'_continue' if args.continue_run else '_warm' if args.load_file else ''}"
        model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    save_every = 10

    def save_callback(model, optimizer, scheduler, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        log_file = f'{base_path}/log/{model_string}.log'
        if epoch == "start":
            print(f"Starting training of model {model_string}")
            return
        try:
            os.makedirs(f"{base_path}/log", exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch} loss {model.losses[-1]} learning_rate {model.learning_rates[-1]}\n')
        except Exception as e:
            print(f'Failed to write to log file {log_file}: {e}')
            
        if epoch != "on_exit":
            mlflow.log_metric(key="wallclock_time", value=model.wallclock_times[-1], step=epoch)
            mlflow.log_metric(key="loss", value=model.losses[-1], step=epoch)
            mlflow.log_metric(key="learning_rate", value=model.learning_rates[-1], step=epoch)
        
        try:
            if (epoch == "on_exit") or epoch % save_every == 0:
                file_name = f'models_diff/{model_string}_epoch_{epoch}.cpkt'
                os.makedirs(f"{base_path}/models_diff", exist_ok=True)
                disk_usage = shutil.disk_usage(f"{base_path}/models_diff")
                if disk_usage.free < 1024 * 1024 * 1024 * 2:
                    print("Not saving model, not enough disk space")
                    print("DISK FULLLLLLL")
                    return
                with open(log_file, 'a') as f:
                    f.write(f'Saving model to {base_path}/{file_name}\n')
                print(f'Saving model to {base_path}/{file_name}')
                config_sample['epoch_in_training'] = epoch
                config_sample['learning_rates'] = model.learning_rates
                config_sample['losses'] = model.losses
                config_sample['wallclock_times'] = model.wallclock_times

                save_model(model, optimizer, scheduler, base_path, file_name, config_sample)
                # remove last checkpoint
                if epoch != "on_exit" and epoch - save_every > 0 and model.losses[-1] < 1:
                    old_file_name = f'{base_path}/models_diff/{model_string}_epoch_{epoch - save_every}.cpkt'
                    if os.path.exists(old_file_name):
                        try:
                            os.remove(old_file_name)
                        except Exception as e:
                            print(f"Failed to remove old model file {old_file_name}: {e}")

        except Exception as e:
            print("WRITING TO MODEL FILE FAILED")
            print(e)

    if socket.gethostname() == "amueller-tabpfn-4gpu":
        mlflow.set_tracking_uri("http://localhost:5000")
    else:            
        mlflow.set_tracking_uri("http://20.114.249.177:5000")

    mlflow.set_experiment(args.experiment)

    if args.continue_run and not args.create_new_run:
        # find run id via mlflow
        run_ids = mlflow.search_runs(filter_string=f"run_name='{model_string}'")['run_id']
        if len(run_ids) > 1:
            raise ValueError(f"Found more than one run with name {model_string}")
        run_id = run_ids.iloc[0]
        run_args = {'run_id': run_id}

    else:
        run_args = {'run_name': model_string}

    with mlflow.start_run(**run_args):
        mlflow.log_param('hostname', socket.gethostname())
        mlflow.log_params({k:v for k, v in config_sample.items() if isinstance(v, (int, float, str)) and k != 'epoch_in_training'})
        total_loss, model, dl = get_model(config_sample
                            , device
                            , should_train=True
                            , verbose=1
                            , epoch_callback=save_callback, model_state=model_state, optimizer_state=optimizer_state, scheduler=scheduler,
                            load_model_strict=args.continue_run or args.load_strict)    

    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return total_loss, model, dl

if __name__ == "__main__":
    main(sys.argv[1:])
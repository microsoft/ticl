import socket
import sys
import time

import mlflow
import torch
import os
from git import Repo

from tabpfn.mlflow_utils import MLFLOW_HOSTNAME
from tabpfn.model_builder import get_model
from tabpfn.model_configs import get_base_config
from tabpfn.utils import compare_dicts, argparser_from_config, init_device, get_model_string, synetune_handle_checkpoint, make_training_callback

def main(argv):
    config = get_base_config()
    parser = argparser_from_config(config)
    args = parser.parse_args(argv)

    device, rank, num_gpus = init_device(args.gpu_id, args.use_cpu)

    # handle syne-tune restarts
    args.base_path, args.continue_run, args.warm_start_from, report = synetune_handle_checkpoint(args)

    if args.create_new_run and not args.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = args.base_path

    torch.set_num_threads(24)
    config.update(vars(args))

    if config['seed_everything']:
        import lightning as L
        L.seed_everything(42)
        
    config['num_gpus'] = 1
    config['device'] = device

    warm_start_weights = args.warm_start_from
    config['nhead'] = config['emsize'] // 128

    config['num_steps'] = args.num_steps or 1024 * 64 // config['batch_size'] // config['aggregate_k_gradients']
    config['weight_embedding_rank'] = args.weight_embedding_rank if args.low_rank_weights else None

    if config['model_type'] == 'perceiver' and config['perceiver_large_dataset']:
        config['max_eval_pos'] = 8 * 1000
        config['n_samples'] = 8 * 1024+128

    if config['extra_fast_test']:
        config['max_eval_pos'] = 16
        config['n_samples'] = 2 * 16
        config['nhead'] = 1

    config['boolean_prior'] = {'max_fraction_uninformative': args.boolean_max_fraction_uninformative, 'p_uninformative': args.boolean_p_uninformative}
    save_every = args.save_every

    model_state, optimizer_state, scheduler = None, None, None
    if warm_start_weights is not None:
        model_state, old_optimizer_state, old_scheduler, old_config = torch.load(
            warm_start_weights, map_location='cpu')
        module_prefix = 'module.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        if args.continue_run:
            config = old_config
            # we want to overwrite specific parts of the old config with current values
            config['device'] = device
            config['warm_start_from'] = warm_start_weights
            optimizer_state = old_optimizer_state
            config['stop_after_epochs'] = args.stop_after_epochs
            if not args.restart_scheduler:
                scheduler = old_scheduler
        else:
            print("WARNING warm starting with new settings")
            compare_dicts(config, old_config, all=True)

    model_string = get_model_string(config, args, parser)
    save_callback = make_training_callback(save_every, model_string, base_path, report, config, args.no_mlflow, args.st_checkpoint_dir)

    if not args.no_mlflow:
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOSTNAME}:5000")

        tries = 0
        while tries < 5:
            try:
                mlflow.set_experiment(args.experiment)
                break
            except:
                tries += 1
                print(f"Failed to set experiment, retrying {tries}/5")
                time.sleep(5)

        if args.continue_run and not args.create_new_run:
            # find run id via mlflow
            run_ids = mlflow.search_runs(filter_string=f"attribute.run_name='{model_string}'")['run_id']
            if len(run_ids) > 1:
                raise ValueError(f"Found more than one run with name {model_string}")
            run_id = run_ids.iloc[0]
            run_args = {'run_id': run_id}

        else:
            run_args = {'run_name': model_string}

        path = os.path.dirname(os.path.abspath(__file__))
        run_args['tags'] = {'mlflow.source.git.commit': Repo(path, search_parent_directories=True).head.object.hexsha}

        with mlflow.start_run(**run_args):
            mlflow.log_param('hostname', socket.gethostname())
            mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str)) and k != 'epoch_in_training'})
            total_loss, model, dl, epoch = get_model(config, device, should_train=True, verbose=1, epoch_callback=save_callback, model_state=model_state,
                                                     optimizer_state=optimizer_state, scheduler=scheduler,
                                                     load_model_strict=args.continue_run or args.load_strict)

    else:
        total_loss, model, dl, epoch = get_model(config, device, should_train=True, verbose=1, epoch_callback=save_callback, model_state=model_state,
                                                 optimizer_state=optimizer_state, scheduler=scheduler,
                                                 load_model_strict=args.continue_run or args.load_strict)

    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return {'loss': total_loss, 'model': model, 'dataloader': dl,
            'config': config, 'base_path': base_path,
            'model_string': model_string, 'epoch': epoch}


if __name__ == "__main__":
    main(sys.argv[1:])

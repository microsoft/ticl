# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import socket
import sys
import time

import mlflow

import torch
import os
root_dir = os.path.dirname(os.path.abspath(__file__))

from git import Repo

from ticl.model_builder import get_model
from ticl.utils import init_device, get_model_string, synetune_handle_checkpoint, make_training_callback
from ticl.config_utils import compare_dicts, flatten_dict, update_config
from ticl.cli_parsing import make_model_level_argparser
from ticl.model_configs import get_model_default_config
from argparse import Namespace

import pandas as pd
import pdb

def main(argv, extra_config=None):
    # extra config is used for testing purposes only
    # this is the generic entry point for training any model, so it has A LOT of options
    parser = make_model_level_argparser()
    args = parser.parse_args(args=argv or ['--help'])
    model = args.ssm.model if 'ssm' in args.model_type else None
    config = get_model_default_config(args.model_type, model)

    device, rank, num_gpus = init_device(args.general.gpu_id, args.general.use_cpu)
    # handle syne-tune restarts
    orchestration = args.orchestration
    orchestration.base_path, orchestration.continue_run, orchestration.warm_start_from, report = synetune_handle_checkpoint(orchestration)

    if orchestration.create_new_run and not orchestration.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = orchestration.base_path
    torch.set_num_threads(24)
    for group_name in vars(args):
        if group_name == "model_type":
            # the only non-group argument from the top level parser
            config['model_type'] = args.model_type
            continue
        if group_name not in config:
            config[group_name] = {}
        for k, v in vars(getattr(args, group_name)).items():
            if isinstance(v, Namespace):
                if k not in config[group_name]:
                    config[group_name][k] = {}
                # FIXME we only allow one level of nesting, we should do recursion here really.
                config[group_name][k].update(vars(v))
            else:
                config[group_name][k] = v
        config[group_name].update()
    if args.orchestration.seed_everything:
        import lightning as L
        L.seed_everything(42)

    if 'transformer' in config:
        attention_type = 'transformer'
    elif 'ssm' in config:
        attention_type = 'ssm'
    else:
        raise ValueError(f"Unknown attention type")

    # promote general group to top level
    config.update(config.pop('general'))
    config['num_gpus'] = 1
    config['device'] = device

    if not config[attention_type]['classification_task']:
        print('Setting regression parameters')
        config['prior']['classification']['max_num_classes'] = 0
        config[attention_type]['y_encoder'] = 'linear'
        config['mothernet']['decoder_type'] = 'average'

    warm_start_weights = orchestration.warm_start_from
    config[attention_type]['nhead'] = config[attention_type]['emsize'] // 128

    config['dataloader']['num_steps'] = config['dataloader']['num_steps'] or 1024 * \
        64 // config['dataloader']['batch_size'] // config['optimizer']['aggregate_k_gradients']

    if args.orchestration.extra_fast_test:
        config['prior']['n_samples'] = 2 * 16
        config[attention_type]['nhead'] = 1

    if extra_config is not None:
        update_config(config, extra_config)

    save_every = orchestration.save_every

    model_state, optimizer_state, scheduler = None, None, None
    if warm_start_weights is not None:
        model_state, old_optimizer_state, old_scheduler, old_config = torch.load(
            warm_start_weights, map_location='cpu')
        module_prefix = 'module.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        if args.orchestration.continue_run:
            config = old_config
            # we want to overwrite specific parts of the old config with current values
            config['device'] = device
            config['orchestration']['warm_start_from'] = warm_start_weights
            config['orchestration']['continue_run'] = True
            optimizer_state = old_optimizer_state
            config['orchestration']['stop_after_epochs'] = args.orchestration.stop_after_epochs
            if not args.orchestration.restart_scheduler:
                scheduler = old_scheduler
        else:
            print("WARNING warm starting with new settings")
            compare_dicts(config, old_config)

    if config['orchestration']['detect_anomaly']:
        print("ENABLING GRADIENT DEBUGGING (detect-anomaly)! Don't use for training.")
        torch.autograd.set_detect_anomaly(True)

    model_string = get_model_string(config, num_gpus, device, parser)
    save_callback = make_training_callback(
        save_every, 
        model_string, 
        base_path, 
        report, 
        config, 
        orchestration.use_mlflow,
        orchestration.st_checkpoint_dir, 
        classification=config[attention_type]['classification_task'], 
        validate=orchestration.validate
    )

    mlflow_hostname = os.environ.get("MLFLOW_HOSTNAME", None)
    if orchestration.use_wandb:
        import wandb
        from ticl.environment import WANDB_INFO
        wandb_data, flatten_key_dict = flatten_dict(config, track_keys=True)
        wandb_config = {k: v for k, v in wandb_data.items() if k not in ['wallclock_times', 'losses', 'learning_rates']}
        # check_keys = pd.read_csv(f"{root_dir}/configs/{args.model_type}_configs.csv").columns
        # flatten_check_keys = [flatten_key_dict[k] for k in check_keys] + ['model_type']

        # api = wandb.Api(timeout=300)
        # runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")
        # find_existing_run = None
        # for run in runs:
        #     run_config_list = {k: v for k,v in run.config.items() if not k.startswith('_')}
        #     this_run = True
        #     for key in flatten_check_keys:
        #         if key not in wandb_config or key not in run_config_list:
        #             this_run = False
        #             break
        #         if (run_config_list[key] != wandb_config[key]):
        #             # check whether they are numbers 
        #             this_run = False
        #             if (isinstance(run_config_list[key], (int, float))) and (isinstance(wandb_config[key], (int, float))):
        #                 # check whether the numbers are close
        #                 if abs(run_config_list[key] - wandb_config[key]) <= 1e-5:
        #                     this_run = True
        #             if not this_run: break
        #     if this_run:

        #         print("########"*3)
        #         print(f"Find existing run in wandb: {run.name}")
        #         print("########"*3)

        #         if not orchestration.wandb_overwrite: 
        #             find_existing_run = run
        #             print(f'wandb_overwrite is set to {orchestration.wandb_overwrite}, exiting...')
        #             exit(0)
                
            
        # # initialize wandb
        # if find_existing_run is None:
        wandb.init(
            dir=WANDB_INFO['dir'],
            project=WANDB_INFO['project'],
            entity=WANDB_INFO['entity'],
            id=model_string,
            config=wandb_config,
        )

    if (not orchestration.use_mlflow) or mlflow_hostname is None:
        print("Not logging run with mlflow, set MLFLOW_HOSTNAME environment to variable enable mlflow.")
        total_loss, model, dl, epoch = get_model(
            config, 
            device, 
            should_train=True,
            verbose=1, 
            epoch_callback=save_callback, 
            model_state=model_state,
            optimizer_state=optimizer_state, 
            scheduler=scheduler,
            load_model_strict=orchestration.continue_run or orchestration.load_strict
        )
    else:
        print(f"Logging run with mlflow at host {mlflow_hostname}")
        mlflow.set_tracking_uri(f"http://{mlflow_hostname}:5000")

        tries = 0
        while tries < 5:
            try:
                mlflow.set_experiment(orchestration.experiment)
                break
            except:
                tries += 1
                print(f"Failed to set experiment, retrying {tries}/5")
                time.sleep(5)

        if orchestration.continue_run and not orchestration.create_new_run:
            # find run id via mlflow
            run_ids = mlflow.search_runs(filter_string=f"attribute.run_name='{model_string}'")['run_id']
            if len(run_ids) > 1:
                raise ValueError(f"Found more than one run with name {model_string}")
            if len(run_ids) < 1:
                raise ValueError(f"Found no run with name {model_string}")
            run_id = run_ids.iloc[0]
            run_args = {'run_id': run_id}

        else:
            run_args = {'run_name': model_string}

        path = os.path.dirname(os.path.abspath(__file__))
        run_args['tags'] = {'mlflow.source.git.commit': Repo(path, search_parent_directories=True).head.object.hexsha}

        with mlflow.start_run(**run_args):
            mlflow.log_param('hostname', socket.gethostname())
            mlflow.log_params({k: v for k, v in flatten_dict(config).items() if k not in ['wallclock_times', 'losses', 'learning_rates']})
            total_loss, model, dl, epoch = get_model(
                config, 
                device, 
                should_train=True, 
                verbose=1, 
                epoch_callback=save_callback, 
                model_state=model_state,
                optimizer_state=optimizer_state, 
                scheduler=scheduler,
                load_model_strict=orchestration.continue_run or orchestration.load_strict
            )

    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return {'loss': total_loss, 'model': model, 'dataloader': dl,
            'config': config, 'base_path': base_path,
            'model_string': model_string, 'epoch': epoch}


if __name__ == "__main__":
    main(sys.argv[1:])

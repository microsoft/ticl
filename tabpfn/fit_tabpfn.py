from datetime import datetime
import os
import torch
import mlflow
import sys
from pathlib import Path
from syne_tune import Reporter
import numpy as np
import time


from tabpfn.scripts.model_builder import get_model, save_model
from tabpfn.scripts.model_configs import evaluate_hypers, get_base_config_paper
from tabpfn.utils import load_model_state, make_base_parser, init_device, get_model_string, init_mlflow

import socket
import shutil

def main(argv):
    config = get_base_config_paper()
    parser = make_base_parser("Train TabPFN")
    args = parser.parse_args(argv)

    # handle syne-tune restarts
    checkpoint_dir = args.st_checkpoint_dir
    if checkpoint_dir is not None:
        args.base_path = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.mothernet"
        if checkpoint_path.exists():
            args.continue_run = True
            args.load_file = checkpoint_path

    device, rank, num_gpus = init_device(args.gui_id, args.use_cpu)

    if args.create_new_run and not args.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = args.base_path

    torch.set_num_threads(24)
    config['num_gpus'] = num_gpus
    args_dict = vars(args)
    config.update(args_dict)

    config['nhead'] = config['emsize'] // 128
    config['num_steps'] = args.num_steps or 1024 * 64 // config['batch_size'] // config['aggregate_k_gradients']

    save_every = args.save_every

    config_sample = evaluate_hypers(config)

    if args.load_file is not None:
        model_state, optimizer_state, scheduler, config = load_model_state(args.load_file, config)

    report = Reporter()

    model_string = get_model_string(config, args, parser)

    run_args = init_mlflow(args.experiment, model_string, args.continue_run and not args.create_new_run)

    with mlflow.start_run(**run_args) as run:
        mlflow.log_param('hostname', socket.gethostname())
        mlflow.log_params({k:v for k, v in config_sample.items() if isinstance(v, (int, float, str)) and k != 'epoch_in_training'})
        total_loss, model, dl, epoch = get_model(config_sample
                            , device
                            , should_train=True
                            , verbose=1
                            , epoch_callback=save_callback, model_state=model_state, optimizer_state=optimizer_state, scheduler=scheduler,
                            load_model_strict=args.continue_run or args.load_strict)

    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return {'loss': total_loss, 'model': model, 'dataloader': dl,
            'config': config, 'base_path': base_path,
            'model_string': model_string, 'epoch': epoch}

if __name__ == "__main__":
    main(sys.argv[1:])
import datetime
import glob
import itertools
import math
import os
import random
import re
import shutil
import socket
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from scipy.signal import convolve
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def get_nan_value(v, set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = (data >= data.unsqueeze(-3))
    x = x.sum(0)
    return x
# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?


def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = (data[:, :, col] >= data[:, :, col].unsqueeze(-2))
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('nan'), set_value_to_nan)


def nan_handling_missing_for_no_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('-inf'), set_value_to_nan)


def nan_handling_missing_for_a_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('inf'), set_value_to_nan)


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def bool_mask_to_att_mask(mask):
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    # print('init dist')
    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")
        return True, rank, f'cuda:{rank}'
    elif 'SLURM_PROCID' in os.environ and torch.cuda.device_count() > 1:
        # this is for multi gpu when starting with submitit
        assert device != 'cpu:0'
        rank = int(os.environ['SLURM_PROCID'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        print('distributed submitit launch and my rank is', rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=20),
                                             world_size=torch.cuda.device_count(), rank=rank)
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
              "only I can print, but when using print(..., force=True) it will print on all ranks.")

        return True, rank, f'cuda:{rank}'
    else:
        # print('Not using distributed')
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, device

# NOP function for python with statements (x = NOP(); with x:)


class NOP():
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, 'num_outputs'):
        print('`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on.')
        assert dl.num_outputs != 1, "We assume num_outputs to be 1. Instead of the num_ouputs change your loss." \
                                    "We specify the number of classes in the CE loss."


def product_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def normalize_by_used_features_f(x, num_features_used, num_features):
    return x / (num_features_used / num_features)


def compare_dicts(left, right, prefix=None, all=False):
    if not all:
        for d in [left, right]:
            d.pop("losses", None)
            d.pop("learning_rates", None)
            d.pop("wallclock_times", None)
            d.pop("n_samples_extra_samples", None)
            d.pop("num_classes", None)
            d.pop("differentiable_hyperparameters", None)
            d.pop("num_features_used", None)

    prefix = prefix or ""
    for k in set(left).union(set(right)):
        if k not in left:
            print(f"{prefix}{k} missing in left")
            continue
        if k not in right:
            print(f"{prefix}{k} missing in right")
            continue
        if isinstance(left[k], dict):
            compare_dicts(left[k], right[k], prefix=f"{prefix}{k}->", all=all)
        else:
            if (torch.is_tensor(left[k]) and (left[k] != right[k]).all()) or (not torch.is_tensor(left[k]) and left[k] != right[k]):
                print(f"{prefix}{k}: left: {left[k]}, right: {right[k]}")

def merge_dicts(*dicts):
    keys = set([k for d in dicts for k in d])
    merged = {}
    for k in keys:
        values = [d[k] for d in dicts if k in d]
        if len(values) == 1:
            merged[k] = values[0]
        elif all([isinstance(v, dict) for v in values]):
            merged[k] = merge_dicts(*values)
        else:
            raise ValueError(f"Can't merge {values} for key {k}")
    return merged


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, min_lr=None, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


class ReduceLROnSpike:
    """Reduce learning rate when a metric has bounced up.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        smoothing (int): Number of epochs with over which to smooth recent performance.
            Default: 10.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        tolerance (int): Multiple of std from recent data to be considered a spike.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, smoothing=10,
                 min_lr=0, verbose=False, tolerance=4, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.smoothing = smoothing
        self.verbose = verbose
        self.mode = mode
        self.eps = eps
        self.tolerance = tolerance
        self.last_epoch = 0
        self.recent_losses = []
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if len(self.recent_losses) < self.smoothing:
            self.recent_losses.append(current)
        else:
            sign = -1 if self.mode == 'min' else 1

            if np.mean(self.recent_losses) < current + self.tolerance * sign * np.std(self.recent_losses):
                if self.verbose:
                    print("That loss looks bad!")
                    print("Recent losses:", self.recent_losses)
                    print("Current loss:", current)
                self._reduce_lr(epoch)
                self.recent_losses = []
            else:
                self.recent_losses = self.recent_losses[1:] + [current]

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print(f'Epoch {epoch_str}: reducing learning rate of group {i} from {old_lr:.4e} to {new_lr:.4e}.')

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr
    

def init_device(gpu_id, use_cpu):
    # Single GPU training, get GPU ID from command line
    if 'LOCAL_RANK' in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print('torch.distributed.launch and my rank is', rank)
        num_gpus = int(os.environ["WORLD_SIZE"])
        raise ValueError("Gave up on multi-gpu for now")
    else:
        rank = 0
        num_gpus = 1

    device = "cuda"
    if gpu_id is not None:
        if use_cpu:
            raise ValueError("Can't use cpu and gpu at the same time")
        device = f'cuda:{gpu_id}'
    elif use_cpu:
        device = 'cpu'
    return device, rank, num_gpus


def get_model_string(args, parser, num_gpus, device):
    if args.orchestration.continue_run:
        if args.orchestration.st_checkpoint_dir is None:
            model_string = args.orchestration.warm_start_from.split("/")[-1].split("_epoch_")[0]
            if args.orchestration.create_new_run:
                model_string = model_string + '_continue_'+datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        else:
            with open(f"{args.orchestration.st_checkpoint_dir }/model_string.txt", 'r') as f:
                model_string = f.read()
    else:
        mm = args.general.model_type
        model_type_string = mm if mm in ["perceiver", "additive"] else ('mn' if mm == "mlp" else "tabpfn")

        default_args_dict = vars(parser.parse_args([]))
        args_dict = vars(args)
        config_string = ""
        for arg in parser._actions:
            if arg.option_strings:
                k = arg.dest
                if k in ['st_checkpoint_dir', 'save_every', 'run_id', 'warm_start_from', 'use_cpu', 'continue_run', 'restart_scheduler', 'load_strict', 'gpu_id', 'help', 'base_path', 'create_new_run', 'experiment', 'model_type'] or k not in args_dict:
                    continue
                v = args_dict[k]
                short_name = arg.option_strings[0].replace('-', '')
                if v != default_args_dict[k]:
                    if isinstance(v, float):
                        config_string += f"_{short_name}{v:.4g}"
                    else:
                        config_string += f"_{short_name}{v}"
        gpu_string = f"_{num_gpus}_gpu{'s' if num_gpus > 1 else ''}" if device != 'cpu' else '_cpu'
        model_string = f"{model_type_string}{config_string}{gpu_string}{'_continue' if args.orchestration.continue_run else '_warm' if args.orchestration.warm_start_from else ''}"
        model_string = model_string + '_'+datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if args.orchestration.st_checkpoint_dir is not None:
            with open(f"{args.orchestration.st_checkpoint_dir}/model_string.txt", 'w') as f:
                f.write(model_string)
    return model_string


def make_training_callback(save_every, model_string, base_path, report, config, no_mlflow, checkpoint_dir):
    from tabpfn.model_builder import save_model
    config = config.copy()
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
            wallclock_ticker = max(1, int(model.wallclock_times[-1]//(60 * 5)))
            if not no_mlflow:
                mlflow.log_metric(key="wallclock_time", value=model.wallclock_times[-1], step=epoch)
                mlflow.log_metric(key="loss", value=model.losses[-1], step=epoch)
                mlflow.log_metric(key="learning_rate", value=model.learning_rates[-1], step=epoch)
                mlflow.log_metric(key="wallclock_ticker", value=wallclock_ticker, step=epoch)
            if report is not None:
                # synetune callback
                report(epoch=epoch, loss=model.losses[-1], wallclock_time=wallclock_ticker)  # every 5 minutes

        try:
            if (epoch == "on_exit") or epoch % save_every == 0:
                if checkpoint_dir is not None:
                    if epoch == "on_exit":
                        return
                    file_name = f'{base_path}/checkpoint.mothernet'
                else:
                    file_name = f'{base_path}/models_diff/{model_string}_epoch_{epoch}.cpkt'
                os.makedirs(f"{base_path}/models_diff", exist_ok=True)
                disk_usage = shutil.disk_usage(f"{base_path}/models_diff")
                if disk_usage.free < 1024 * 1024 * 1024 * 2:
                    print("Not saving model, not enough disk space")
                    print("DISK FULLLLLLL")
                    return
                with open(log_file, 'a') as f:
                    f.write(f'Saving model to {file_name}\n')
                print(f'Saving model to {file_name}')
                config['epoch_in_training'] = epoch
                config['learning_rates'] = model.learning_rates
                config['losses'] = model.losses
                config['wallclock_times'] = model.wallclock_times

                save_model(model, optimizer, scheduler, base_path, file_name, config)
                # remove checkpoints that are worse than current
                if epoch != "on_exit" and epoch - save_every > 0:
                    this_loss = model.losses[-1]
                    for i in range(epoch // save_every):
                        loss = model.losses[i * save_every - 1]  # -1 because we start at epoch 1
                        old_file_name = f'{base_path}/models_diff/{model_string}_epoch_{i * save_every}.cpkt'
                        if os.path.exists(old_file_name):
                            if loss > this_loss:
                                try:
                                    print(f"Removing old model file {old_file_name}")
                                    os.remove(old_file_name)
                                except Exception as e:
                                    print(f"Failed to remove old model file {old_file_name}: {e}")
                            else:
                                print(f"Not removing old model file {old_file_name} because loss is too high ({loss} < {this_loss})")

        except Exception as e:
            print("WRITING TO MODEL FILE FAILED")
            print(e)
    return save_callback


def load_model_state(load_path, config):
    model_state, old_optimizer_state, old_scheduler, old_config = torch.load(
        load_path, map_location='cpu')
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    if config['continue_run']:
        config_sample = old_config
        config_sample['device'] = config['device']
        config_sample['warm_start_from'] = load_path
        optimizer_state = old_optimizer_state
        config_sample['stop_after_epochs'] = config['stop_after_epochs']
        if not config['restart_scheduler']:
            scheduler = old_scheduler
    else:
        print("WARNING warm starting with new settings")
        compare_dicts(config_sample, old_config, all=True)

    return model_state, optimizer_state, scheduler, config_sample


def init_mlflow(experiment_name, model_string, continue_run):
    if socket.gethostname() == "amueller-tabpfn-4gpu":
        mlflow.set_tracking_uri("http://localhost:5000")
    else:
        mlflow.set_tracking_uri("http://20.114.249.177:5000")

    tries = 0
    while tries < 5:
        try:
            mlflow.set_experiment(experiment_name)
            break
        except:
            tries += 1
            print(f"Failed to set experiment, retrying {tries}/5")
            time.sleep(5)

    if continue_run:
        # find run id via mlflow
        run_ids = mlflow.search_runs(filter_string=f"attribute.run_name='{model_string}'")['run_id']
        if len(run_ids) > 1:
            raise ValueError(f"Found more than one run with name {model_string}")
        run_id = run_ids.iloc[0]
        run_args = {'run_id': run_id}

    else:
        run_args = {'run_name': model_string}


def synetune_handle_checkpoint(args):
    # handle syne-tune restarts
    checkpoint_dir = args.st_checkpoint_dir
    base_path = args.base_path
    warm_start_from = args.warm_start_from
    continue_run = args.continue_run
    report = None
    if checkpoint_dir is not None:
        from syne_tune import Reporter
        report = Reporter()
        base_path = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.mothernet"
        if checkpoint_path.exists():
            continue_run = True
            warm_start_from = checkpoint_path
    return base_path, continue_run, warm_start_from, report

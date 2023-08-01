import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext


import torch
from torch import nn

import tabpfn.utils as utils

from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr
from tabpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn


def get_criterion(max_num_classes):
    if max_num_classes == 2:
        loss = nn.BCEWthLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = nn.CrossEntropyLoss(reduction='none', weight=torch.ones(max_num_classes))
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def eval_criterion(criterion, targets, output, device, n_out):
    if isinstance(criterion, nn.GaussianNLLLoss):
        assert output.shape[-1] == 2, \
                        'need to write a little bit of code to handle multiple regression targets at once'

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        losses = criterion(output.flatten(), targets.to(device).flatten())
    elif isinstance(criterion, nn.CrossEntropyLoss):
        losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
    else:
        losses = criterion(output, targets)
    losses = losses.view(*output.shape[0:2])
    return utils.torch_nanmean(losses.mean(0), return_nanshare=True)

def train_epoch(model, aggregate_k_gradients, using_dist, scaler, dl, device, optimizer, criterion, n_out):
        model.train()  # Turn on the train mode
        total_loss = 0.
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        steps_per_epoch = len(dl)
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                with autocast(enabled=scaler is not None):
                    output = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                                   , single_eval_pos=single_eval_pos)

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    loss, nan_share = eval_criterion(criterion, targets, output, device=device, n_out=n_out)
                    loss = loss / aggregate_k_gradients

                if scaler: loss = scaler.scale(loss)
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if torch.isnan(loss):
                    print("NAN loss encountered")
                else:
                    total_loss += loss.mean().cpu().detach().item()
                nan_steps += nan_share
                ignore_steps += (targets == -100).float().mean()

            before_get_batch = time.time()
        return (total_loss / steps_per_epoch * aggregate_k_gradients,
                time_to_get_batch, forward_time, step_time, nan_steps.cpu().item() / steps_per_epoch,
                ignore_steps.cpu().item()/steps_per_epoch)


def train(dl, model, criterion, optimizer_state=None, scheduler=None,
          epochs=10, lr=None, weight_decay=0.0, warmup_epochs=10,
          validation_period=10, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, epoch_callback=None, train_mixed_precision=False, adaptive_batch_size=False,
          ):
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    using_dist, rank, device = init_dist(device)
    if rank == 0:
        print(f'Using {device} device')

    model.to(device)
    criterion.to(device)

    n_out = model.n_out
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        if rank == 0:
            print("Distributed training")
    elif "cuda" in device:
        print(f"Single GPU training on {torch.cuda.get_device_name()}")
    elif "cpu" in device:
        pass
    else:
        raise ValueError(f"Invalid device: {device}")

    if rank == 0:
        model.learning_rates = getattr(model, 'learning_rates', [])
        model.losses = getattr(model, 'losses', [])
        model.wallclock_times = getattr(model, 'wallclock_times', [])
        model.start_time = time.time()
        if len(model.wallclock_times):
            model.start_time -= model.wallclock_times[-1]
        if epoch_callback is not None:
            epoch_callback(model, None, None, "start")


    dl.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        if rank == 0:
            print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is None:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
        start_epoch = 1
    else:
        start_epoch = scheduler.last_epoch + 1
    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    total_loss = float('inf')
    increased_batch_size = 0

    try:
        for epoch in range(start_epoch, epochs + 1):
            if verbose:
                print(f"start of epoch {epoch}")

            epoch_start_time = time.time()
            new_loss, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch(model, aggregate_k_gradients, using_dist, scaler, dl, device, optimizer, criterion, n_out)
            if new_loss > 1.5 * total_loss:
                print("LOSS DIVERGED")
            total_loss = new_loss
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            last_lr = scheduler.get_last_lr()[0]

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.4f} | ')

                print(
                    f' lr {last_lr} data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' 
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)
            if adaptive_batch_size:
                if increased_batch_size == 0 and total_loss <= .55:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 1
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 1 and total_loss <= .50:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 2
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 2 and total_loss <= .45:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 3
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
            scheduler.step()
            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                model.learning_rates.append(last_lr)
                model.losses.append(total_loss)
                model.wallclock_times.append(time.time() - model.start_time)
                epoch_callback(model, optimizer, scheduler, epoch)
            

    except KeyboardInterrupt:
        pass

    if rank == 0: # trivially true for non-parallel training
        return total_loss, model.to('cpu'), dl

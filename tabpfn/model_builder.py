import math
import os
import subprocess as sp
from functools import partial

import torch

import tabpfn.models.encoders as encoders
from tabpfn.assemble_model import assemble_model
from tabpfn.dataloader import get_dataloader
from tabpfn.train import get_criterion, train
from tabpfn.model_configs import get_base_config

try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None

    import cloudpickle
    torch.save((model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


@cache
def load_model(path, device, verbose=False):
    states = torch.load(path, map_location='cpu')
    model_state = states[0]
    config_sample = states[-1]
    if 'y_encoder' not in config_sample and 'onehot' in path:
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def get_encoder(config):
    if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
            ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if 'encoder' in config and config['encoder'] == 'featurewise_mlp':
        encoder = encoders.FeaturewiseMLP
    return encoder


def get_y_encoder(config):
    if config['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['max_num_classes'], emsize=config['emsize'])
    elif config['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config['y_encoder']}")
    return y_encoder


def get_model(config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None, scheduler=None, epoch_callback=None, load_model_strict=True):
    # copy config. Maybe should be a deepcopy?
    passed_config = config.copy()
    config = get_base_config()
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    criterion = get_criterion(config['max_num_classes'])

    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config:
        config['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config:
        config['y_encoder'] = 'linear'
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            config['model_type'] = config['model_maker']
        else:
            config['model_type'] = 'tabpfn'

    epochs = 0 if not should_train else config['epochs']

    dl = get_dataloader(config=config, steps_per_epoch=config['num_steps'], batch_size=config['batch_size'], n_samples=config['n_samples'], device=device,
                        prior_type=config['prior_type'])
    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)
    model = assemble_model(encoder_generator=encoder, y_encoder=y_encoder, num_features=config['num_features'], emsize=config['emsize'], nhead=config['nhead'],
                           nhid=config['emsize'] * config['nhid_factor'], nlayers=config['nlayers'], dropout=config['dropout'],
                           input_normalization=config['input_normalization'],  model_type=config['model_type'], max_num_classes=config['max_num_classes'],
                           predicted_hidden_layer_size=config['predicted_hidden_layer_size'],
                           model_state=model_state, load_model_strict=load_model_strict,
                           decoder_embed_dim=config['decoder_embed_dim'], decoder_two_hidden_layers=config['decoder_two_hidden_layers'],
                           decoder_hidden_size=config['decoder_hidden_size'], no_double_embedding=config['no_double_embedding'],
                           verbose=verbose_train, pre_norm=config['pre_norm'], efficient_eval_masking=config['efficient_eval_masking'],
                           output_attention=config['output_attention'], predicted_hidden_layers=config['predicted_hidden_layers'],
                           special_token=config['special_token'], weight_embedding_rank=config['weight_embedding_rank'] if config['low_rank_weights'] else None,
                           num_latents=config['num_latents'], input_bin_embedding=config['input_bin_embedding'], factorized_output=config['factorized_output'], output_rank=config['output_rank'],
                           bin_embedding_rank=config['bin_embedding_rank'], low_rank_weights=config['low_rank_weights'])

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    model = train(dl,
                  model, criterion=criterion,
                  optimizer_state=optimizer_state, scheduler=scheduler, epochs=epochs, stop_after_epochs=config['stop_after_epochs'],
                  warmup_epochs=config['warmup_epochs'], device=device, aggregate_k_gradients=config['aggregate_k_gradients'], epoch_callback=epoch_callback,
                  learning_rate=config['lr'], min_lr=config['min_lr'],
                  learning_rate_schedule=config['learning_rate_schedule'], lr_decay=config['lr_decay'], verbose=verbose_train, train_mixed_precision=config['train_mixed_precision'],
                  weight_decay=config['weight_decay'], adaptive_batch_size=config['adaptive_batch_size'],
                  reduce_lr_on_spike=config['reduce_lr_on_spike'], adam_beta1=config['adam_beta1'], spike_tolerance=config['spike_tolerance']
                  )

    return model

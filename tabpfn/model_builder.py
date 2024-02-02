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

def get_model(config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None, scheduler=None, epoch_callback=None, load_model_strict=True):
    # copy config. Maybe should be a deepcopy?
    passed_config = config.copy()
    config = get_base_config()
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    criterion = get_criterion(config['prior']['classification']['max_num_classes'])
    
    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config['prior']:
        config['prior']['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config['transformer']:
        config['transformer']['y_encoder'] = 'linear'
    if 'model_type' not in passed_config['general']:
        config['general']['model_type'] = 'tabpfn'

    config['mothernet']['low_rank_weights'] = passed_config['mothernet'].get('low_rank_weights', passed_config['mothernet'].get('weight_embedding_rank', None) is not None)

    epochs = 0 if not should_train else config['optimizer']['epochs']

    dl = get_dataloader(prior_config=config['prior'], optimizer_config=config['optimizer'], diff_config=config['differentiable_hyperparameters'], device=device)
    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)
    model = assemble_model(encoder=encoder, y_encoder_layer=y_encoder, model_type=config['general']['model_type'], config_transformer=config['transformer'],
                           config_mothernet=config['mothernet'], config_additive=config['additive'], config_perceiver=config['perceiver'],
                           num_features=config['prior']['num_features'], max_num_classes=config['prior']['classification']['max_num_classes'])
    
    if model_state is not None:
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    model = train(dl,
                  model, criterion=criterion,
                  optimizer_state=optimizer_state, scheduler=scheduler, epochs=epochs, epoch_callback=epoch_callback, verbose=verbose_train, **config['optimizer'])

    return model

from functools import partial
import tabpfn.encoders as encoders

from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_uniform_single_eval_pos_sampler
from tabpfn.dataloader import get_dataloader
from tabpfn.assemble_model import assemble_model
from tabpfn.train import train, get_criterion

import torch

import subprocess as sp
import os
import math

try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    config_sample = {**config_sample}
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

    _, model, _ = get_model(config_sample, device=device, should_train=False, verbose=verbose)
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
    if 'y_encoder' not in config:
        config['y_encoder'] = 'one_hot'
    if config['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['max_num_classes'], emsize=config['emsize'])
    elif config['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config['y_encoder']}")
    return y_encoder


def get_model(config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None, scheduler=None, epoch_callback=None, load_model_strict=True):
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    if 'aggregate_k_gradients' not in config or config['aggregate_k_gradients'] is None:
        config['aggregate_k_gradients'] = math.ceil(config['batch_size'] * ((config['nlayers'] * config['emsize'] * config['bptt'] * config['bptt']) / 10824640000))

    criterion = get_criterion(config['max_num_classes'])

    # DEFAULTS
    config['multiclass_type'] = config.get('multiclass_type', 'rank')
    config['mix_activations'] = config.get('mix_activations', False)
    config['recompute_attn'] = config.get('recompute_attn', False)
    config['weight_decay'] = config.get('weight_decay', 0.0)
    config['pre_norm'] = config.get('pre_norm', False)
    config['decoder_embed_dim'] = config.get('decoder_embed_dim', 2048)
    config['predicted_hidden_layer_size'] = config.get('predicted_hidden_layer_size', None)
    config['predicted_hidden_layers'] = config.get('predicted_hidden_layers', 1)
    config['weight_embedding_rank'] = config.get('weight_embedding_rank', None)
    config['learning_rate_schedule'] = config.get('learning_rate_schedule', 'cosine')
    config['warmup_epochs'] = config.get('warmup_epochs', 20)

    config['eval_positions'] = [int(config['bptt'] * 0.95)]
    model_maker = config.get('model_maker', False)
    epochs = 0 if not should_train else config['epochs']


    dataloader_config = dict(steps_per_epoch=config['num_steps'], batch_size=config['batch_size'], bptt=config['bptt'], device=device,
                             prior_type=config['prior_type'], flexible=config['flexible'], differentiable=config['differentiable'],
                             single_eval_pos_gen=get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']),
                                                                                     min_len=config.get('min_eval_pos', 0)),)
    dl = get_dataloader(config=config,
                        **dataloader_config)
    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)
    model = assemble_model(encoder_generator=encoder, y_encoder=y_encoder, num_features=config['num_features'], emsize=config['emsize'], nhead=config['nhead'],
                           nhid=config['emsize'] * config['nhid_factor'], nlayers=config['nlayers'], dropout=config['dropout'],
                           input_normalization=config.get('input_normalization', False),  model_maker=model_maker, max_num_classes=config['max_num_classes'],
                           predicted_hidden_layer_size=config['predicted_hidden_layer_size'],
                           model_state=model_state, load_model_strict=load_model_strict,
                           decoder_embed_dim=config['decoder_embed_dim'], decoder_two_hidden_layers=config.get('decoder_two_hidden_layers', False),
                           decoder_hidden_size=config.get('decoder_hidden_size', None), no_double_embedding=config.get('no_double_embedding', False),
                           verbose=True, pre_norm=config['pre_norm'], efficient_eval_masking=config.get('efficient_eval_masking', False),
                           output_attention=config.get('output_attention', False), predicted_hidden_layers=config['predicted_hidden_layers'],
                           special_token=config.get('special_token', False), weight_embedding_rank=config['weight_embedding_rank'],)
    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    model = train(dl,
                  model, criterion=criterion,
                  optimizer_state=optimizer_state, scheduler=scheduler
                  , epochs=epochs
                  , warmup_epochs=config['warmup_epochs']
                  , gpu_device=device
                  , aggregate_k_gradients=config['aggregate_k_gradients']
                  , epoch_callback=epoch_callback
                  , lr=config['lr']
                  , learning_rate_schedule=config['learning_rate_schedule']
                  , verbose=verbose_train,
                  weight_decay=config['weight_decay'], adaptive_batch_size=config.get('adaptive_batch_size', False))

    return model

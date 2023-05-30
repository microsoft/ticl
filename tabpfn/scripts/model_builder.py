from functools import partial
import tabpfn.encoders as encoders

from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_uniform_single_eval_pos_sampler
from tabpfn.dataloader import get_dataloader
from tabpfn.assemble_model import assemble_model

import torch

import subprocess as sp
import os
import math

def save_model(model, path, filename, config_sample):
    config_sample = {**config_sample}

    import cloudpickle
    torch.save((model.state_dict(), None, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)



def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


def load_model_only_inference(path, filename, device, verbose=False):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    model_state, optimizer_state, config_sample = torch.load(os.path.join(path, filename), map_location='cpu')

    if (('nan_prob_no_reason' in config_sample and config_sample['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config_sample and config_sample['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config_sample and config_sample['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if 'encoder' in config_sample and config_sample['encoder'] == 'featurewise_mlp':
        encoder = encoders.FeaturewiseMLP

    n_out = config_sample['max_num_classes']

    device = device if torch.cuda.is_available() else 'cpu:0'
    encoder = encoder(config_sample['num_features'], config_sample['emsize'])

    nhid = config_sample['emsize'] * config_sample['nhid_factor']

    assert config_sample['max_num_classes'] > 2
    loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.ones(int(config_sample['max_num_classes'])))

    if 'y_encoder' not in config_sample:
        if 'onehot' in filename:
            config_sample['y_encoder'] = 'one_hot'
        else:
            config_sample['y_encoder'] = 'linear'

    if config_sample['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config_sample['max_num_classes'], emsize=config_sample['emsize'])
    elif config_sample['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config_sample['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config_sample['y_encoder']}")

    model = TransformerModel(encoder, n_out, config_sample['emsize'], config_sample['nhead'], nhid,
                             config_sample['nlayers'], y_encoder=y_encoder,
                             dropout=config_sample['dropout'],
                             efficient_eval_masking=config_sample['efficient_eval_masking'])
    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float('inf'), float('inf'), model), config_sample # no loss measured

def load_model(path, filename, device, verbose=False):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    print('!! Warning: GPyTorch must be installed !!')
    model_state, _, config_sample = torch.load(
        os.path.join(path, filename), map_location='cpu')
    if ('differentiable_hyperparameters' in config_sample
            and 'prior_mlp_activations' in config_sample['differentiable_hyperparameters']):
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values_used'] = config_sample[
                                                                                                         'differentiable_hyperparameters'][
                                                                                                         'prior_mlp_activations'][
                                                                                                         'choice_values']
        config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values'] = [
            torch.nn.Tanh for k in config_sample['differentiable_hyperparameters']['prior_mlp_activations']['choice_values']]

    config_sample['categorical_features_sampler'] = lambda: lambda x: ([], [], [])
    config_sample['num_features_used_in_training'] = config_sample['num_features_used']
    config_sample['num_features_used'] = lambda: config_sample['num_features']
    # config_sample['num_classes_in_training'] = config_sample['num_classes']
    # config_sample['num_classes'] = 2
    # config_sample['batch_size_in_training'] = config_sample['batch_size']
    # config_sample['batch_size'] = 1
    # config_sample['bptt_in_training'] = config_sample['bptt']
    # config_sample['bptt'] = 10
    config_sample['bptt_extra_samples_in_training'] = config_sample['bptt_extra_samples']
    config_sample['bptt_extra_samples'] = None

    model = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model[2].load_state_dict(model_state)
    model[2].to(device)
    model[2].eval()

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


def get_model(config, device, should_train=True, verbose=False, state_dict=None, epoch_callback=None, load_model_strict=True):
    from tabpfn.train import train, Losses
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    if 'aggregate_k_gradients' not in config or config['aggregate_k_gradients'] is None:
        config['aggregate_k_gradients'] = math.ceil(config['batch_size'] * ((config['nlayers'] * config['emsize'] * config['bptt'] * config['bptt']) / 10824640000))


    if config['max_num_classes'] == 2:
        loss = Losses.bce
    elif config['max_num_classes'] > 2:
        loss = Losses.ce(config['max_num_classes'])
    else:
        raise ValueError(f"Invalid number of classes: {config['max_num_classes']}")

    # DEFAULTS
    config['multiclass_type'] = config['multiclass_type'] if 'multiclass_type' in config else 'rank'
    config['mix_activations'] = config['mix_activations'] if 'mix_activations' in config else False
    config['recompute_attn'] = config['recompute_attn'] if 'recompute_attn' in config else False

    config['bptt_extra_samples'] = config['bptt_extra_samples'] if 'bptt_extra_samples' in config else None
    config['eval_positions'] = [int(config['bptt'] * 0.95)] if config['bptt_extra_samples'] is None else [int(config['bptt'])]

    model_maker = config.get('model_maker', False)
    epochs = 0 if not should_train else config['epochs']


    dataloader_config = dict(steps_per_epoch=config['num_steps'], batch_size=config['batch_size'], bptt=config['bptt'],
                             bptt_extra_samples=config['bptt_extra_samples'], device=device,
                             single_eval_pos_gen=get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']),
                                                                                     min_len=config.get('min_eval_pos', 0)))
    dl = get_dataloader(config['prior_type'], config['flexible'], config['differentiable'], config,
                        **dataloader_config)
    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)
    model = assemble_model(encoder_generator=encoder, y_encoder=y_encoder, num_features=config['num_features'], emsize=config['emsize'], nhead=config['nhead'],
                           nhid=config['emsize'] * config['nhid_factor'], nlayers=config['nlayers'], dropout=config['dropout'],
                           input_normalization=config.get('input_normalization', False),  model_maker=model_maker, criterion=loss,
                           predicted_hidden_layer_size=config.get('predicted_hidden_layer_size', None),
                           load_weights_from_this_state_dict=state_dict, load_model_strict=load_model_strict)
    model = train(dl,
                  model,
                  criterion=loss
                  , epochs=epochs
                  , warmup_epochs=20
                  , bptt=config['bptt']
                  , gpu_device=device
                  , steps_per_epoch=config['num_steps']
                  , single_eval_pos_gen=get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']), min_len=config.get('min_eval_pos', 0))
                  , aggregate_k_gradients=config['aggregate_k_gradients']
                  , epoch_callback=epoch_callback
                  , bptt_extra_samples = config['bptt_extra_samples']
                  , lr=config['lr']
                  , verbose=verbose_train,
                  weight_decay=config.get('weight_decay', 0.0))

    return model

import torch

from tabpfn.priors.utils import uniform_int_sampler_f
from collections import defaultdict

def get_general_config(max_features, n_samples):
    """"
    Returns the general PFN training hyperparameters.
    """
    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        "max_eval_pos": n_samples,
        "n_samples_used": n_samples,
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
        "mix_activations": False,  # False means to mix activations
        "pre_sample_causes": True,
        "multiclass_type": 'rank'}
    
    optimizer = {
        "aggregate_gradients": 1,
        "batch_size": 8,
        "learing_rate": 0.00003,
        "epochs": 4000,
        "num_steps": None,
        "epochs": 80,
    }

    transformer = {
        "em_size": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
    }

    return {'prior': prior, 'optimizer': optimizer, 'transformer': transformer}


def get_flexible_categorical_config(max_features):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    max_num_classes = 10
    config_flexible_categorical = {
        "nan_prob_unknown_reason_reason_prior": 0.5,
        "nan_prob_a_reason": 0.0,
        "max_num_classes": max_num_classes,
        "num_classes": uniform_int_sampler_f(2, max_num_classes),
        "noise_type": "Gaussian",  # NN
        "balanced": False,
        "num_features_used":
            {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(1, max_features)}
    }
    return {'prior': config_flexible_categorical}


def get_diff_gp():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        #'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
        #                'lower_bound': 0},
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        #'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
        #                'lower_bound': 0},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
    }

    return diff_gp


def get_diff_causal():
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
        # This mustn't be too high since activations get too large otherwise
        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
        #"noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
        #              'lower_bound': 0.0},
        #"init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
        #             'lower_bound': 0.0},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                       'lower_bound': 2},

        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},

        # 'pre_sample_causes': {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal


def get_diff_prior_bag():
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
        # MLP Weight (Biased, since MLP works better, 1.0 is weight for prior number 0)
    }

    return diff_prior_bag


def get_diff_config():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    diff_prior_bag = get_diff_prior_bag()
    diff_causal = get_diff_causal()
    diff_gp = get_diff_gp()

    config_diff = {'differentiable_hyperparameters': {**diff_prior_bag, **diff_causal, **diff_gp}}

    return config_diff


def get_prior_config_causal(max_features=100):
    config_general = get_general_config(max_features, n_samples=1024+128)
    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_diff = get_diff_config()

    # config = {'general': config_general, 'flexible_categorical': config_flexible_categorical, 'diff': config_diff}
    config = {**config_general, **config_flexible_categorical, **config_diff}
    return config


def get_base_config():
    config = get_prior_config_causal()
    config = defaultdict(dict, config)
    # prior
    config['prior'].update({
        'heterogeneous_batches': False,
        'add_uninformative_features': False,
        'recompute_attn': True,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        'pre_sample_causes': True,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_unknown_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'boolean_max_fraction_uninformative': 0.5,
        'boolean_p_uninformative': 0.5,
        'set_value_to_nan': .1,
        'prior_mlp_scale_weights_sqrt': True,
        'random_feature_rotation': True
    })

    config['model-type'] = 'mothernet'

    config['mothernet'].update({
        'weight_embedding_rank': None,
        'predicted_hidden_layer_size': 128,
        'output_attention': True,
        'decoder_embed_dim': 2048,
        'predicted_hidden_layers': 1,
        'decoder_two_hidden_layers': False,
        'decoder_hidden_size': None,
        'no_double_embedding': True,
        'special_token': False})

    config['perceiver']['num_latents'] = 512

    config['additive']['shared_embedding'] = False
    
    config['transformer'].update({
        'pre_norm': False,
        'y_encoder': "one_hot",
        'efficient_eval_masking': True,
        'hid_factor': 2,
        'input_normalization': False
    })

    # training
    config['training'].update({
        'stop_after_epochs': None,
        'reduce_lr_on_spike': False,
        'warmup_epochs': 20,
        'learning_rate_schedule': 'cosine',
        'min_eval_pos': 2,
        'max_eval_pos': 1000,
        'min_lr': None,
        'adam_beta1': 0.9,
        'spike_tolerance': 4,
        'weight_decay': 0.0,
        'lr_decay': 0.99,
        'adaptive_batch_size': True
    })
    return config

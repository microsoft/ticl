import torch

from mothernet.distributions import uniform_int_sampler_f
from mothernet.config_utils import merge_dicts


def get_general_config(max_features, n_samples):
    """"
    Returns the general PFN training hyperparameters.
    """
    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        'prior_type': 'prior_bag',
        'prior_bag': {'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0}}}

    mlp_prior_config = {"pre_sample_causes": True,
                        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
                        'prior_mlp_scale_weights_sqrt': True,
                        'random_feature_rotation': True,
                        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
                        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
                        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                        # This mustn't be too high since activations get too large otherwise
                        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
                        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
                        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                       'lower_bound': 2},
                        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
                            torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU
                        ]},
                        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}

    prior['mlp'] = mlp_prior_config

    gp_prior_config = {
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]},
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
    }

    prior['gp'] = gp_prior_config

    dataloader = {
        "batch_size": 8,
        "num_steps": None,
        'min_eval_pos': 2,
        'max_eval_pos': 1000}

    optimizer = {
        "aggregate_k_gradients": 1,
        "learning_rate": 0.00003,
        "epochs": 4000,
        "train_mixed_precision": True

    }

    transformer = {
        "emsize": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
        'nhead': 512 // 128

    }

    return {'prior': prior, 'optimizer': optimizer, 'transformer': transformer, 'dataloader': dataloader}


def get_classification_prior_config(max_features, n_samples):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    max_num_classes = 10
    config_classsification_prior = {
        "nan_prob_unknown_reason_reason_prior": 0.5,
        "nan_prob_a_reason": 0.0,
        "max_num_classes": max_num_classes,
        "num_classes": uniform_int_sampler_f(2, max_num_classes),
        # "noise_type": "Gaussian",  # NN unused?!
        "balanced": False,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        "multiclass_type": 'rank',
        "num_features_used": uniform_int_sampler_f(1, max_features),
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_unknown_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': .1,

    }
    return {'prior': {'classification': config_classsification_prior}}


def get_prior_config_causal(max_features=100):
    config_general = get_general_config(max_features, n_samples=1024+128)
    config_classsification_prior = get_classification_prior_config(max_features, n_samples=1024+128)
    config = merge_dicts(config_general, config_classsification_prior)
    return config


def get_base_config():
    config = get_prior_config_causal()
    config['prior'].update({
        'heterogeneous_batches': False,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'boolean': {
            'max_fraction_uninformative': 0.5,
            'p_uninformative': 0.5},
    })

    config['model_type'] = 'mothernet'

    config['mothernet'] = {
        'weight_embedding_rank': None,
        'low_rank_weights': True,
        'predicted_hidden_layer_size': 128,
        'output_attention': True,
        'decoder_embed_dim': 1024,
        'predicted_hidden_layers': 1,
        'decoder_two_hidden_layers': False,
        'decoder_hidden_size': 2048,
        'special_token': False}

    config['perceiver'] = {'num_latents': 512}

    config['additive'] = {
        'input_bin_embedding': 'none',
        'factorized_output': False,
        'output_rank': 16,
        'bin_embedding_rank': 16}

    config['transformer'].update({
        'recompute_attn': True,
        'pre_norm': False,
        'y_encoder': "one_hot",
        'efficient_eval_masking': True,
        'input_normalization': False
    })

    config['optimizer'].update({
        'stop_after_epochs': None,
        'reduce_lr_on_spike': False,
        'warmup_epochs': 20,
        'learning_rate_schedule': 'cosine',
        'min_lr': 1e-8,
        'adam_beta1': 0.9,
        'spike_tolerance': 4,
        'weight_decay': 0.0,
        'lr_decay': 0.99,
        'adaptive_batch_size': True
    })
    return config

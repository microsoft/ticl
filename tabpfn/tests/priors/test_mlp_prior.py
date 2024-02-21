from tabpfn.priors import MLPPrior
from tabpfn.model_configs import get_base_config
import lightning as L
import torch
import pytest

@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_mlp_prior(batch_size, num_features, n_samples):
    # test the mlp prior
    L.seed_everything(42)
    hyperparameters = {
        'pre_sample_causes': True,
        'prior_mlp_activations': torch.nn.ReLU,
        'is_causal' : False,
        'num_causes': 3, # actually ignored because is_causal is False
        'prior_mlp_hidden_dim': 128,
        'num_layers': 3,
        'noise_std': 0.1,
        'y_is_effect': True,
        'pre_sample_weights': False,
        'prior_mlp_dropout_prob': 0.1,
        'block_wise_dropout': True,
        'init_std': 0.1,
        'sort_features': False,
        'in_clique': False,
        'verbose': False,
        'prior_mlp_scale_weights_sqrt': True,
        'random_feature_rotation': True,
        'add_uninformative_features': False,
    }
    mlp_config = dict(prior_mlp_scale_weights_sqrt=hyperparameters['prior_mlp_scale_weights_sqrt'], random_feature_rotation=hyperparameters['random_feature_rotation'],
                                                              pre_sample_causes=hyperparameters['pre_sample_causes'], add_uninformative_features=hyperparameters['add_uninformative_features'])
    prior = MLPPrior(mlp_config)

    x, y, y_ = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu', hyperparameters=hyperparameters)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    if n_samples == 128 and batch_size == 4 and num_features == 11:
        assert float(x[0, 0, 0])== 1.0522834062576294
        assert float(y[0, 0]) == -0.1148308664560318


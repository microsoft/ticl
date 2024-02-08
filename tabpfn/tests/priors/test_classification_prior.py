from tabpfn.model_configs import get_base_config
import lightning as L
import torch
import pytest

from tabpfn.priors import ClassificationAdapterPrior, MLPPrior
from tabpfn.priors.flexible_categorical import ClassificationAdapter
from tabpfn.priors.utils import uniform_int_sampler_f

@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_classification_prior_no_sampling(batch_size, num_features, n_samples, n_classes):
    # test the mlp prior
    L.seed_everything(42)
    config = get_base_config()
    config['prior']['classification']['num_features_used'] = num_features # always using all features in this test
    config['prior']['classification']['num_classes'] = n_classes
    prior = ClassificationAdapterPrior(MLPPrior(config['prior']['mlp']), **config['prior']['classification'])
    hyperparameters = {
        'prior_mlp_activations': lambda: torch.nn.ReLU, # relu is callable so we'd try to call it thinking it's a sampling function....
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
    }
    x, y, y_ = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu', hyperparameters=hyperparameters)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    # because of the strange sampling, we an have less than n_classes many classes.
    assert y_.max() < n_classes
    assert y_.min() == 0
    if n_samples == 128 and batch_size == 4 and num_features == 11 and n_classes == 2:
        assert float(x[0, 0, 0])== -0.46619218587875366
        assert float(y[0, 0]) == 1.0


def test_classification_adapter_with_sampling():
    batch_size = 16
    num_features = 100
    n_samples = 900
    n_classes = 8
    # test the mlp prior
    L.seed_everything(42)
    config = get_base_config()
    hyperparameters = {
        'prior_mlp_activations': lambda: torch.nn.ReLU, # relu is callable so we'd try to call it thinking it's a sampling function....
        'is_causal' : False,
        'num_causes': 3, # actually ignored because is_causal is False
        'prior_mlp_hidden_dim': 128,
        'num_layers': uniform_int_sampler_f(1, 10),
        'noise_std': 0.1,
        'y_is_effect': True,
        'pre_sample_weights': False,
        'prior_mlp_dropout_prob': 0.1,
        'block_wise_dropout': True,
        'init_std': 0.1,
        'sort_features': False,
        'in_clique': False,
    }

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), hyperparameters=hyperparameters, config=config['prior']['classification'])
    assert adapter.h['num_layers'] == 4
    assert adapter.h['num_features_used'] == 8
    assert adapter.h['num_classes'] == 10

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_ = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)

    assert float(x[0, 0, 0])== 0.21882277727127075
    assert float(y[0, 0]) == 0.0

from ticl.model_configs import get_prior_config
import lightning as L
import torch
import pytest
import numpy as np

from ticl.priors import ClassificationAdapterPrior, MLPPrior
from ticl.priors.classification_adapter import ClassificationAdapter


@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_classification_prior_no_sampling(batch_size, num_features, n_samples, n_classes):
    # test the mlp prior
    L.seed_everything(43)
    config = get_prior_config()
    config['prior']['classification']['num_features_used'] = num_features  # always using all features in this test
    config['prior']['classification']['num_classes'] = n_classes
    hyperparameters = {
        'prior_mlp_activations': torch.nn.ReLU,
        'is_causal': False,
        'num_causes': 3,  # actually ignored because is_causal is False
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
    config['prior']['mlp'].update(hyperparameters)

    prior = ClassificationAdapterPrior(MLPPrior(config['prior']['mlp']), **config['prior']['classification'])

    x, y, y_, info = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    # because of the strange sampling, we an have less than n_classes many classes.
    assert y_.max() < n_classes
    assert y_.min() == 0
    if n_samples == 128 and batch_size == 4 and num_features == 11 and n_classes == 2:
        assert float(x[0, 0, 0]) == 1.4561537504196167
        assert float(y[0, 0]) == 1.0


def test_classification_adapter_with_sampling():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=config['prior']['classification'])
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)

    assert float(x[0, 0, 0]) == pytest.approx(-1.6891261339187622)
    assert float(y[0, 0]) == 3.0


def test_classification_adapter_curriculum():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    classification_config = config['prior']['classification']
    classification_config['feature_curriculum'] = True
    classification_config['pad_zeros'] = False

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=classification_config)
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features, 'epoch': 0}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, 1)
    args['epoch'] = 1
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, 1)
    args['epoch'] = 100
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, 51)


def test_classification_adapter_double_sampler():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    classification_config = config['prior']['classification']
    classification_config['num_features_sampler'] = 'double_sample'
    classification_config['pad_zeros'] = False

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=classification_config)
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features, 'epoch': 0}
    num_features = np.array([adapter(batch_size=batch_size, **args)[0].shape[-1] for i in range(10)])
    assert num_features.min() == 5
    assert num_features.max() == 61
    assert (num_features < 20).sum() == 5


def test_classification_adapter_with_sampling_no_padding():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']['classification']
    prior_config['pad_zeros'] = False
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, 72)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)

    assert float(x[0, 0, 0]) == pytest.approx(-1.2161709070205688)
    assert float(y[0, 0]) == 3.0


def test_classification_adapter_nan():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(12)
    config = get_prior_config()
    prior_config = config['prior']['classification']
    prior_config['pad_zeros'] = False
    prior_config['nan_prob_no_reason'] = 0.99
    prior_config['nan_prob_a_reason'] = 0
    prior_config['set_value_to_nan'] = 0.99

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, _ = adapter(batch_size=batch_size, **args)
    assert y.shape == (n_samples, batch_size)
    assert x.isnan().float().mean() > 0.95

    prior_config['nan_prob_no_reason'] = 0
    prior_config['nan_prob_a_reason'] = 0.99
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, _ = adapter(batch_size=batch_size, **args)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    assert x.isnan().float().mean() > 0.45
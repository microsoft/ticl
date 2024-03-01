from mothernet.model_configs import get_base_config
import lightning as L
import torch
import pytest

from mothernet.priors import ClassificationAdapterPrior, MLPPrior
from mothernet.priors.classification_adapter import ClassificationAdapter


@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_classification_prior_no_sampling(batch_size, num_features, n_samples, n_classes):
    # test the mlp prior
    L.seed_everything(42)
    config = get_base_config()
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

    x, y, y_ = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    # because of the strange sampling, we an have less than n_classes many classes.
    assert y_.max() < n_classes
    assert y_.min() == 0
    if n_samples == 128 and batch_size == 4 and num_features == 11 and n_classes == 2:
        assert float(x[0, 0, 0]) == -0.46619218587875366
        assert float(y[0, 0]) == 1.0


def test_classification_adapter_with_sampling():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_base_config()
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=config['prior']['classification'])
    # assert adapter.h['num_layers'] == 6
    # assert adapter.h['num_features_used'] == 7
    # assert adapter.h['num_classes'] == 3

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_ = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)

    assert float(x[0, 0, 0]) == pytest.approx(0.6690560579299927)
    assert float(y[0, 0]) == 2.0

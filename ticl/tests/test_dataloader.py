from ticl.dataloader import get_dataloader
from ticl.model_configs import get_prior_config
from ticl.priors import BagPrior, ClassificationAdapterPrior
from ticl.distributions import LogUniformHyperparameter

import lightning as L

import pytest


def test_get_dataloader_base_config():
    L.seed_everything(42)
    config = get_prior_config()
    # config['num_causes'] = 3
    # config['num_features'] = 10
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    batch_size = 16
    dataloader_config['batch_size'] = batch_size
    n_samples = 1024
    prior_config['n_samples'] = n_samples
    n_features = 100
    prior_config['num_features'] = n_features
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, device="cpu")
    # calling get_batch explicitly means we have to repeate some paramters but then we can look at the sampled hyperparameters
    prior = dataloader.prior
    assert isinstance(prior, BagPrior)
    assert isinstance(prior.base_priors['gp'], ClassificationAdapterPrior)
    assert isinstance(prior.base_priors['mlp'], ClassificationAdapterPrior)
    mlp_prior_config = prior.base_priors['mlp'].base_prior.config
    assert isinstance(mlp_prior_config['noise_std'], LogUniformHyperparameter)
    assert mlp_prior_config['noise_std'].min == 1e-4
    assert mlp_prior_config['noise_std'].max == 0.5
    assert mlp_prior_config['noise_std']() == 0.002428916946974888
    assert dataloader.prior.prior_weights == {'mlp': 0.961, 'gp': 0.039}
    x, y, y_, info = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu")

    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
    # assert config_sample['num_layers'].alpha == 0.6722902794233997
    # assert config_sample['num_layers'].scale == 2.497327922401265
    # assert config_sample['prior_bag_exp_weights_1'] == 3.4672360788274705
    # assert config_sample['is_causal'] == True
    # assert config_sample['sort_features'] == False
    # assert config_sample['noise_std'] == 0.016730402817820244

    assert (x[:, :, :] == 0).reshape(-1, x.shape[-1]).all(axis=0).int().argmax() == 61

    x, y, y_, info = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu")
    assert (x[:, :, :] == 0).reshape(-1, x.shape[-1]).all(axis=0).int().argmax() == 83
    # assert config_sample['noise_std'] == 0.0004896957955177838
    # assert config_sample['sort_features'] == True
    # assert config_sample['is_causal'] == False


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("n_samples", [7, 256, 512, 2200])
@pytest.mark.parametrize("n_features", [5, 15, 100, 200, 311])
@pytest.mark.parametrize("prior_type", ["prior_bag", "boolean_only", "bag_boolean"])
def test_get_dataloader_parameters_passed(batch_size, n_samples, n_features, prior_type):
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['num_steps'] = 1
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features'] = n_features
    prior_config['prior_type'] = prior_type
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, device="cpu")
    (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_no_nan_in_flexible():
    # this apparently doesn't run long enough to find the occasional nan
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['num_steps'] = 100
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, device="cpu")
    for i, ((_, x, y), target_y, single_eval_pos) in enumerate(dataloader):
        assert not x.isnan().any()


def test_get_dataloader_nan_in_flexible(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['num_steps'] = 1
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features'] = n_features
    prior_class = prior_config['classification']
    prior_class['nan_prob_a_reason'] = .5
    prior_class['nan_prob_no_reason'] = .5
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_uninformative_mlp(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features'] = n_features
    prior_config['mlp']['add_uninformative_features'] = True

    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)

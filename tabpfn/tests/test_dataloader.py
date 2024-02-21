from tabpfn.dataloader import get_dataloader
from tabpfn.model_configs import get_base_config
from tabpfn.priors import SamplerPrior, BagPrior, ClassificationAdapterPrior
from tabpfn.priors.differentiable_prior import LogUniformHyperparameter

import lightning as L

import pytest

def test_get_dataloader_base_config():
    L.seed_everything(42)
    config = get_base_config()
    # config['num_causes'] = 3
    # config['num_features'] = 10
    # num_features really doesn't work lol
    batch_size = 16
    n_samples = 1024
    n_features = 100
    config['num_features'] = n_features
    dataloader = get_dataloader(prior_type="prior_bag", config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    # calling get_batch explicitly means we have to repeate some paramters but then we can look at the sampled hyperparameters
    prior = dataloader.prior
    assert isinstance(prior, SamplerPrior)
    assert isinstance(prior.base_prior, BagPrior)
    assert isinstance(prior.base_prior.base_priors['gp'], ClassificationAdapterPrior)
    assert isinstance(prior.base_prior.base_priors['mlp'], ClassificationAdapterPrior)
    assert set(prior.hyper_dists.keys()) == set(['prior_bag_exp_weights_1', 'num_layers', 'prior_mlp_hidden_dim', 'prior_mlp_dropout_prob', 'init_std', 'noise_std', 'num_causes', 'is_causal', 'pre_sample_weights', 'y_is_effect', 'prior_mlp_activations',
                                                 'block_wise_dropout', 'sort_features', 'in_clique', 'outputscale', 'lengthscale', 'noise'])
    assert prior.hyper_dists['prior_bag_exp_weights_1'].max == 10
    assert isinstance(prior.hyper_dists['noise_std'], LogUniformHyperparameter)
    assert prior.hyper_dists['noise_std'].min == 1e-4
    assert prior.hyper_dists['noise_std'].max == 0.5
    assert prior.hyper_dists['noise_std']() == 0.002428916946974888
    assert dataloader.prior.base_prior.prior_weights == {'mlp': 0.961, 'gp': 0.039}

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)

    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
    assert config_sample['num_layers'].alpha == 0.6722902794233997
    assert config_sample['num_layers'].scale == 2.497327922401265
    assert config_sample['prior_bag_exp_weights_1'] == 3.4672360788274705
    assert config_sample['is_causal'] == True
    assert config_sample['sort_features'] == False
    assert config_sample['noise_std'] == 0.016730402817820244

    assert (x[:, :, 60:] == 0).all()
    assert (x[:, :, 59] != 0).all()

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert (x[:, :, 66] == 0).all()
    assert (x[:, :, 65] != 0).all()
    assert config_sample['noise_std'] == 0.0004896957955177838
    assert config_sample['sort_features'] == True
    assert config_sample['is_causal'] == False


def test_get_dataloader_heterogeneous_batches():
    L.seed_everything(42)
    config = get_base_config()
    config['heterogeneous_batches'] = True
    # config['num_causes'] = 3
    # config['num_features'] = 10
    # num_features really doesn't work lol
    batch_size = 16
    n_samples = 1024
    n_features = 100
    config['num_features'] = n_features
    dataloader = get_dataloader(prior_type="prior_bag", config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    # calling get_batch explicitly means we have to repeate some paramters but then we can look at the sampled hyperparameters
    prior = dataloader.prior
    assert isinstance(prior, SamplerPrior)
    assert isinstance(prior.base_prior, BagPrior)
    assert isinstance(prior.base_prior.base_priors['gp'], ClassificationAdapterPrior)
    assert isinstance(prior.base_prior.base_priors['mlp'], ClassificationAdapterPrior)
    assert set(prior.hyper_dists.keys()) == set(['prior_bag_exp_weights_1', 'num_layers', 'prior_mlp_hidden_dim', 'prior_mlp_dropout_prob', 'init_std', 'noise_std', 'num_causes', 'is_causal', 'pre_sample_weights', 'y_is_effect', 'prior_mlp_activations',
                                                 'block_wise_dropout', 'sort_features', 'in_clique', 'outputscale', 'lengthscale', 'noise'])
    assert prior.hyper_dists['prior_bag_exp_weights_1'].max == 10
    assert isinstance(prior.hyper_dists['noise_std'], LogUniformHyperparameter)
    assert prior.hyper_dists['noise_std'].min == 1e-4
    assert prior.hyper_dists['noise_std'].max == 0.5
    assert prior.hyper_dists['noise_std']() == 0.002428916946974888
    assert dataloader.prior.base_prior.prior_weights == {'mlp': 0.961, 'gp': 0.039}

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)

    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
    alphas = [x['num_layers'].alpha for x in config_sample]
    assert alphas[-1] == 0.1283677857195334
    assert alphas[0] == 0.6722902794233997
    scales = [x['num_layers'].alpha for x in config_sample]
    assert scales[-1] == 0.1283677857195334
    assert scales[0] == 0.6722902794233997
    assert config_sample[-1]['prior_bag_exp_weights_1'] == 8.114981297773888
    assert config_sample[-1]['is_causal'] == False
    assert config_sample[-1]['sort_features'] == False
    assert config_sample[-1]['noise_std'] == 0.001965810983472645
    assert len(config_sample) == batch_size

    # 98 features
    assert (x[:, :, :] == 0).reshape(-1, x.shape[-1]).all(axis=0).int().argmax() == 98

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert (x[:, :, :] == 0).reshape(-1, x.shape[-1]).all(axis=0).int().argmax() == 97
    assert config_sample[-1]['noise_std'] == 0.23350879018430812
    assert config_sample[-1]['sort_features'] == True
    assert config_sample[-1]['is_causal'] == True



@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("n_samples", [256, 512])
@pytest.mark.parametrize("n_features", [100, 200, 311])
@pytest.mark.parametrize("prior_type", ["prior_bag", "boolean_only", "bag_boolean"])
def test_get_dataloader_parameters_passed(batch_size, n_samples, n_features, prior_type):
    L.seed_everything(42)
    config = get_base_config()
    config['num_features'] = n_features
    # we shouldn't use these parameters from the config here, only what was explicitly passed
    config.pop("n_samples")
    dataloader = get_dataloader(prior_type=prior_type, config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_nan_in_flexible(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_base_config()
    config['nan_prob_a_reason'] = .5
    config['nan_prob_no_reason'] = .5
    config['nan_prob_unknown_reason'] = .5
    config['nan_prob_unknown_reason_reason_prior'] = .5
    config['num_features'] = n_features
    # we shouldn't use these parameters from the config here, only what was explicitly passed
    config.pop("n_samples")
    dataloader = get_dataloader(prior_type="prior_bag", config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_uninformative_mlp(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_base_config()
    config['add_uninformative_features'] = True
    config['num_features'] = n_features
    # we shouldn't use these parameters from the config here, only what was explicitly passed
    config.pop("n_samples")
    dataloader = get_dataloader(prior_type="prior_bag", config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
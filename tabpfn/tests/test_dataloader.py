from tabpfn.dataloader import get_dataloader
from tabpfn.model_configs import get_base_config_paper

import lightning as L

import pytest

def test_get_dataloader_base_config():
    L.seed_everything(42)
    config = get_base_config_paper()
    # config['num_causes'] = 3
    # config['num_features'] = 10
    # num_features really doesn't work lol
    batch_size = 16
    n_samples = 1024
    n_features = 100
    config['num_features'] = n_features
    dataloader = get_dataloader(prior_type="prior_bag", config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    # calling get_batch explicitly means we have to repeate some paramters but then we can look at the sampled hyperparameters
    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
    assert config_sample['prior_bag_exp_weights_1'] == 4.9963209507789
    assert config_sample['is_causal'] == True
    assert config_sample['sort_features'] == True
    assert (x[:, :, 60] == 0).all()
    assert (x[:, :, 59] != 0).all()
    assert config_sample['num_layers']() == 3
    assert config_sample['num_layers']() == 4
    
    assert config_sample['noise_std']() == 0.08150998232279336
    assert config_sample['noise_std']() == 0.10754045266680022

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert (x[:, :, 59] == 0).all()
    assert (x[:, :, 58] != 0).all()
    assert config_sample['noise_std']() == 0.0019014857504532474
    assert config_sample['noise_std']() == 0.0017815365399049398
    assert config_sample['sort_features'] == False
    assert config_sample['is_causal'] == False


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("n_samples", [256, 512])
@pytest.mark.parametrize("n_features", [100, 200, 311])
@pytest.mark.parametrize("prior_type", ["prior_bag", "boolean_only", "bag_boolean"])
def test_get_dataloader_parameters_passed(batch_size, n_samples, n_features, prior_type):
    L.seed_everything(42)
    config = get_base_config_paper()
    config['num_features'] = n_features
    # we shouldn't use these parameters from the config here, only what was explicitly passed
    config.pop("n_samples")
    dataloader = get_dataloader(prior_type=prior_type, config=config, steps_per_epoch=1, batch_size=batch_size, n_samples=n_samples, device="cpu")
    (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_nan_in_flexible(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_base_config_paper()
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
    config = get_base_config_paper()
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
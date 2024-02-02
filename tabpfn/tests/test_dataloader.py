from tabpfn.dataloader import get_dataloader
from tabpfn.model_configs import get_base_config

import lightning as L

import pytest

def test_get_dataloader_base_config():
    L.seed_everything(42)
    config = get_base_config()
    # config['num_causes'] = 3
    # config['num_features'] = 10
    # num_features really doesn't work lol
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    batch_size = 16
    dataloader_config['steps_per_epoch'] = 1
    dataloader_config['batch_size'] = batch_size
    n_samples = 1024
    prior_config['n_samples'] = n_samples
    n_features = 100
    prior_config['num_features']  = n_features
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, diff_config=config['differentiable_hyperparameters'], device="cpu")
    # calling get_batch explicitly means we have to repeate some paramters but then we can look at the sampled hyperparameters
    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
    assert config_sample['prior_bag_exp_weights_1'] == 4.9963209507789
    assert config_sample['is_causal'] == False
    assert config_sample['sort_features'] == False
    assert (x[:, :, 46] == 0).all()
    assert (x[:, :, 45] != 0).all()
    assert config_sample['num_layers']() == 5
    assert config_sample['num_layers']() == 3
    assert config_sample['noise_std'] == 0.016730402817820244

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=batch_size, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert (x[:, :, 52] == 0).all()
    assert (x[:, :, 51] != 0).all()
    assert config_sample['noise_std'] == 0.0036156294364456955
    assert config_sample['sort_features'] == True
    assert config_sample['is_causal'] == True


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("n_samples", [256, 512])
@pytest.mark.parametrize("n_features", [100, 200, 311])
@pytest.mark.parametrize("prior_type", ["prior_bag", "boolean_only", "bag_boolean"])
def test_get_dataloader_parameters_passed(batch_size, n_samples, n_features, prior_type):
    L.seed_everything(42)
    config = get_base_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['steps_per_epoch'] = 1
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features']  = n_features
    prior_config['prior_type'] = prior_type
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, diff_config=config['differentiable_hyperparameters'], device="cpu")
    (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_nan_in_flexible(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_base_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['steps_per_epoch'] = 1
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features'] = n_features
    prior_class = prior_config['classification']
    prior_class['nan_prob_a_reason'] = .5
    prior_class['nan_prob_no_reason'] = .5
    prior_class['nan_prob_unknown_reason'] = .5
    prior_class['nan_prob_unknown_reason_reason_prior'] = .5
    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, diff_config=config['differentiable_hyperparameters'], device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)


def test_get_dataloader_uninformative_mlp(batch_size=16, n_samples=256, n_features=111):
    L.seed_everything(42)
    config = get_base_config()
    prior_config = config['prior']
    dataloader_config = config['dataloader']
    dataloader_config['steps_per_epoch'] = 1
    dataloader_config['batch_size'] = batch_size
    prior_config['n_samples'] = n_samples
    prior_config['num_features'] = n_features
    prior_config['mlp']['add_uninformative_features'] = True

    dataloader = get_dataloader(prior_config=prior_config, dataloader_config=dataloader_config, diff_config=config['differentiable_hyperparameters'], device="cpu")
    for i in range(10):
        # sample a couple times to explore different code paths
        (_, x, y), target_y, single_eval_pos = dataloader.gbm()
    assert x.shape == (n_samples, batch_size, n_features)
    assert y.shape == (n_samples, batch_size)
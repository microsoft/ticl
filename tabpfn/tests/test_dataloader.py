from tabpfn.dataloader import get_dataloader
from tabpfn.model_configs import get_base_config_paper

import lightning as L


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
    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=16, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    assert x.shape == (1024, 16, 100)
    assert y.shape == (1024, 16)
    assert config_sample['prior_bag_exp_weights_1'] == 4.9963209507789
    assert config_sample['is_causal'] == True
    assert config_sample['sort_features'] == True

    assert config_sample['num_layers']() == 3
    assert config_sample['num_layers']() == 4
    
    assert config_sample['noise_std']() == 0.08150998232279336
    assert config_sample['noise_std']() == 0.10754045266680022

    x, y, y_, config_sample = dataloader.prior.get_batch(batch_size=16, n_samples=n_samples, num_features=n_features, device="cpu", hyperparameters=dataloader.hyperparameters)
    
    assert config_sample['noise_std']() == 0.0019014857504532474
    assert config_sample['noise_std']() == 0.0017815365399049398
    assert config_sample['sort_features'] == False
    assert config_sample['is_causal'] == False


# def test_get_dataloader_no_config():
#     dataloader = get_dataloader(prior_type="prior_bag", config={}, steps_per_epoch=1, batch_size=16, n_samples=1024, device="cpu")
#     batch = dataloader.prior.get_batch()
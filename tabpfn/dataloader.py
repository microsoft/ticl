import numpy as np
from torch.utils.data import DataLoader

import tabpfn.priors as priors
from tabpfn.priors.flexible_categorical import FlexibleCategoricalPrior
from tabpfn.priors.prior_bag import BagPrior
from tabpfn.priors.boolean_conjunctions import BooleanConjunctionPrior
from tabpfn.priors.differentiable_prior import SamplerPrior

class PriorDataLoader(DataLoader):
    def __init__(self, prior, num_steps, batch_size, min_eval_pos, max_eval_pos, n_samples, device, num_features, hyperparameters):
        self.prior = prior
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos
        self.max_eval_pos = max_eval_pos
        self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.hyperparameters = hyperparameters
        self.epoch_count = 0

    def gbm(self, epoch=None):
        # Actually can only sample up to max_eval_pos-1 but that's how it was in the original code
        single_eval_pos = np.random.randint(self.min_eval_pos, self.max_eval_pos)
        batch = self.prior.get_batch(batch_size=self.batch_size, n_samples=self.n_samples, num_features=self.num_features, device=self.device,
                                     hyperparameters=self.hyperparameters, epoch=epoch,
                                     single_eval_pos=single_eval_pos)
        # we return sampled hyperparameters from get_batch for testing but we don't want to use them as style.
        x, y, target_y, _ = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
        return (None, x, y), target_y, single_eval_pos
    
    def __len__(self):
        return self.num_steps

    def get_test_batch(self):  # does not increase epoch_count
        return self.gbm(epoch=self.epoch_count)

    def __iter__(self):
        self.epoch_count += 1
        return iter(self.gbm(epoch=self.epoch_count - 1) for _ in range(self.num_steps))


def get_dataloader(prior_type, config, steps_per_epoch, batch_size, n_samples, device):
    gp_flexible = FlexibleCategoricalPrior(priors.fast_gp.GPPrior())
    mlp_flexible = FlexibleCategoricalPrior(priors.mlp.MLPPrior())

    hyperparameters = config.copy()
    if 'num_features_used' in hyperparameters:
        hyperparameters['num_features_used'] = config['num_features_used']['uniform_int_sampler_f(3,max_features)']
    
    if prior_type == 'prior_bag':
        # Prior bag combines priors
        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                             prior_weights={'mlp': 0.961, 'gp': 0.039})
        prior = SamplerPrior(base_prior=bag_prior, differentiable_hyperparameters=config['differentiable_hyperparameters'])
    elif prior_type == "boolean_only":
        prior = BooleanConjunctionPrior()
    elif prior_type == "bag_boolean":
        boolean = BooleanConjunctionPrior()
        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'boolean': boolean},
                             prior_weights={'mlp': 0.9, 'gp': 0.02, 'boolean': 0.08})
        prior = SamplerPrior(base_prior=bag_prior, differentiable_hyperparameters=config['differentiable_hyperparameters'])
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")
    
    # fixme get rid of passing whole config as hyperparameters here
    return PriorDataLoader(prior=prior, num_steps=steps_per_epoch, batch_size=batch_size, n_samples=n_samples, min_eval_pos=config['min_eval_pos'],
                           max_eval_pos=config['max_eval_pos'], device=device,
                           num_features=config['num_features'], hyperparameters=hyperparameters)

import tabpfn.priors as priors
from tabpfn.priors.flexible_categorical import FlexibleCategoricalPrior
from tabpfn.priors.prior_bag import BagPrior
from tabpfn.priors.boolean_conjunctions import BooleanConjunctionSampler
from tabpfn.priors.utils import PriorDataLoader
from tabpfn.priors.differentiable_prior import DifferentiableSamplerPrior

def get_dataloader(prior_type, config, steps_per_epoch, batch_size, n_samples, device):
    gp_flexible = FlexibleCategoricalPrior(priors.fast_gp.GPPrior())
    mlp_flexible = FlexibleCategoricalPrior(priors.mlp.MLPPrior())

    hyperparameters = config.copy()
    hyperparameters['num_features_used'] = config['num_features_used']['uniform_int_sampler_f(3,max_features)']
    
    if prior_type == 'prior_bag':
        # Prior bag combines priors
        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                             prior_weights={'mlp': 0.961, 'gp': 0.039})
        prior = DifferentiableSamplerPrior(base_prior=bag_prior, differentiable_hyperparameters=config['differentiable_hyperparameters'])
    elif prior_type == "boolean_only":
        prior = BooleanConjunctionSampler()
    elif prior_type == "bag_boolean":
        boolean = BooleanConjunctionSampler()
        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'boolean': boolean},
                             prior_weights={'mlp': 0.9, 'gp': 0.02, 'boolean': 0.08})
        prior = DifferentiableSamplerPrior(base_prior=bag_prior, differentiable_hyperparameters=config['differentiable_hyperparameters'])
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")
    
    # fixme get rid of passing whole config as hyperparameters here
    return PriorDataLoader(prior=prior, num_steps=steps_per_epoch, batch_size=batch_size, n_samples=n_samples, min_eval_pos=config['min_eval_pos'],
                           max_eval_pos=config['max_eval_pos'], device=device,
                           num_features=config['num_features'], hyperparameters=hyperparameters)

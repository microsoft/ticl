import tabpfn.priors as priors
from tabpfn.priors.flexible_categorical import FlexibleCategoricalPrior
from tabpfn.priors.prior_bag import BagPrior
from tabpfn.priors.boolean_conjunctions import BooleanConjunctionSampler
from tabpfn.priors.utils import PriorDataLoader
from tabpfn.priors.differentiable_prior import DifferentiableSamplerPrior

def get_dataloader(prior_type, config, steps_per_epoch, batch_size, single_eval_pos_gen, n_samples, device):
    gp_flexible = FlexibleCategoricalPrior(priors.fast_gp.GPPrior())
    mlp_flexible = FlexibleCategoricalPrior(priors.mlp.MLPPrior())
    
    if prior_type == 'prior_bag':
        # Prior bag combines priors
        hyperparameters = config.copy()
        hyperparameters['num_features_used'] = config['num_features_used']['uniform_int_sampler_f(3,max_features)']
        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                             prior_exp_weights={'mlp': config['differentiable_hyperparameters']['prior_bag_exp_weights_1']}, verbose=True)
        prior = DifferentiableSamplerPrior(base_prior=bag_prior, differentiable_hyperparameters=config['differentiable_hyperparameters'])

        extra_prior_kwargs_dict = {
            'num_features': config['num_features'], 'hyperparameters': hyperparameters, 'batch_size_per_prior_sample': config.get('batch_size_per_prior_sample', None)
        }
    elif prior_type == "boolean_only":
        prior = BooleanConjunctionSampler()
        extra_prior_kwargs_dict = {
            'num_features': config['num_features'], 'hyperparameters': {}, 'batch_size_per_prior_sample': config.get('batch_size_per_prior_sample', None)
        }
    elif prior_type == "bag_boolean":
        raise NotImplementedError()
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_n_samples_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, n_samples

    return PriorDataLoader(prior=prior, num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_n_samples_sampler=eval_pos_n_samples_sampler, device=device, **extra_prior_kwargs_dict)

import tabpfn.priors as priors
from tabpfn.priors.flexible_categorical import FlexibleCategoricalPrior
from tabpfn.priors.prior_bag import BagPrior

def get_mlp_prior_hyperparameters(config):
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}
    return config


def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}


def get_dataloader(prior_type, config, steps_per_epoch, batch_size, single_eval_pos_gen, bptt, device):
    gp_flexible = FlexibleCategoricalPrior(priors.fast_gp.GPPrior())
    mlp_flexible = FlexibleCategoricalPrior(priors.mlp.MLPPrior())
    
    if prior_type == 'prior_bag':
        # Prior bag combines priors
        prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config)}

        prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config['prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
        prior_hyperparameters['rotate_normalized_labels'] = config['rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

        bag_prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                             prior_exp_weights={'mlp': config['differentiable_hyperparameters']['prior_bag_exp_weights_1']}, verbose=True)
        extra_kwargs = {'get_batch': bag_prior.get_batch, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
        #extra_kwargs = {'get_batch': priors.prior_bag.get_batch_bag, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
        DataLoader = priors.differentiable_prior.DataLoader

        extra_prior_kwargs_dict = {
            # , 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
            'num_features': config['num_features'], 'hyperparameters': prior_hyperparameters, 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None), **extra_kwargs
        }
    elif prior_type == "boolean_only":
        DataLoader = priors.boolean_conjunctions.DataLoader
        extra_prior_kwargs_dict = {
            'num_features': config['num_features'], 'hyperparameters': {}, 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None)
        }
    elif prior_type == "bag_boolean":
        raise NotImplementedError()
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, bptt

    return DataLoader(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
                      seq_len_maximum=bptt, device=device, **extra_prior_kwargs_dict)

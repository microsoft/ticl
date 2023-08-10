
def get_mlp_prior_hyperparameters(config):
    from tabpfn.priors.utils import gamma_sampler_f
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


def get_gp_mix_prior_hyperparameters(config):
    return {'lengthscale_concentration': config["prior_lengthscale_concentration"],
            'nu': config["prior_nu"],
            'outputscale_concentration': config["prior_outputscale_concentration"],
            'categorical_data': config["prior_y_minmax_norm"],
            'y_minmax_norm': config["prior_lengthscale_concentration"],
            'noise_concentration': config["prior_noise_concentration"],
            'noise_rate': config["prior_noise_rate"]}

def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}


def make_get_batch(model_proto, **extra_kwargs):
    def new_get_batch(batch_size, seq_len, num_features, hyperparameters
            , device, model_proto=model_proto
            , **kwargs):
        kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
        return model_proto.get_batch(
            batch_size=batch_size
            , seq_len=seq_len
            , device=device
            , hyperparameters=hyperparameters
            , num_features=num_features, **kwargs)
    return new_get_batch


def get_dataloader(prior_type, flexible, differentiable, config, steps_per_epoch, batch_size,
                   single_eval_pos_gen, bptt, device):
    import tabpfn.priors as priors

    extra_kwargs = {}
    if prior_type != 'prior_bag':
        raise NotImplementedError("This was removed")

    # Prior bag combines priors
    get_batch_gp = make_get_batch(priors.fast_gp)
    get_batch_mlp = make_get_batch(priors.mlp)
    if flexible:
        get_batch_gp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_gp})
        get_batch_mlp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_mlp})
    prior_bag_hyperparameters = {'prior_bag_get_batch': (get_batch_gp, get_batch_mlp)
        , 'prior_bag_exp_weights_1': 2.0}
    prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config)
        , **prior_bag_hyperparameters}
    model_proto = priors.prior_bag


    if flexible:
        prior_hyperparameters['normalize_labels'] = True
        prior_hyperparameters['check_is_compatible'] = True
    prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config['prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
    prior_hyperparameters['rotate_normalized_labels'] = config['rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

    if differentiable:
        get_batch_base = make_get_batch(model_proto, **extra_kwargs)
        extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
        model_proto = priors.differentiable_prior


    extra_prior_kwargs_dict={
                'num_features': config['num_features']
                , 'hyperparameters': prior_hyperparameters
                #, 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
                , 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None)
                , **extra_kwargs
            }
    

    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, bptt

    return model_proto.DataLoader(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
                                 seq_len_maximum=bptt, device=device, **extra_prior_kwargs_dict)
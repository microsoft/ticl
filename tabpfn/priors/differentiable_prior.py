import math

import torch
from torch import nn

from tabpfn.utils import default_device

from .utils import (beta_sampler_f, gamma_sampler_f,
                    trunc_norm_sampler_f, uniform_sampler_f)


def make_gamma(alpha, scale, do_round, lower_bound):
    if do_round:
        return lambda alpha=alpha, scale=scale: lower_bound + round(gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))())
    else:
        return lambda alpha=alpha, scale=scale: lower_bound + gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))()


def meta_trunc_norm_log_scaled(log_mean, log_std, do_round, lower_bound):
    if do_round:
        return lambda: lower_bound + round(trunc_norm_sampler_f(math.exp(log_mean), math.exp(log_mean)*math.exp(log_std))())
    else:    
        return lambda: lower_bound + trunc_norm_sampler_f(math.exp(log_mean), math.exp(log_mean)*math.exp(log_std))()


def make_trunc_norm(mean, std, do_round, lower_bound):
    if do_round:
        return lambda: lower_bound + round(trunc_norm_sampler_f(mean, std)())
    else:
        return lambda: lower_bound + trunc_norm_sampler_f(mean, std)()
    

def make_choice_mixed(*, choice_values, **choices):
    weights = torch.softmax(torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float), 0)  # create a tensor of weights

    def sample():
        s = torch.multinomial(weights, 1, replacement=True).numpy()[0]
        return choice_values[s]()
    return lambda: sample
    

def make_choice(*, choice_values, **choices):
    choices = torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float)
    weights = torch.softmax(choices, 0)  # create a tensor of weights
    sample = torch.multinomial(weights, 1, replacement=True).numpy()[0]
    return choice_values[sample]
            
        
def sample_meta(f, hparams, **kwargs):
    passed = {hp: hparams[hp]() for hp in hparams}
    meta_passed = f(**passed, **kwargs)
    return meta_passed

class UniformHyperparameter:
    def __init__(self, name, min, max):
        self.min = min
        self.max = max
        self.name = name
    def __call__(self):
        return uniform_sampler_f(self.min, self.max)()

class MetaBetaHyperparameter:
    def __init__(self, name, min, max, scale):
        self.scale = scale
        self.name = name
        self.b = UniformHyperparameter('b', min=min, max=max)
        self.k = UniformHyperparameter('k', min=min, max=max)

    def __call__(self):
        return beta_sampler_f(a=self.b(), b=self.k(), scale=self.scale)


class MetaGammaHyperparameter:
    def __init__(self, name, max_alpha, max_scale, lower_bound, round):
        self.name = name
        self.alpha = UniformHyperparameter('alpha', min=0, max=math.log(max_alpha))  # Surprise Logarithm!
        self.scale = UniformHyperparameter('scale', min=0, max=max_scale)
        self.lower_bound = lower_bound
        self.round = round

    def __call__(self):
        return sample_meta(make_gamma, {"alpha": self.alpha, "scale": self.scale}, lower_bound=self.lower_bound, do_round=self.round)
    
class MetaTruncNormLogScaledHyperparameter:
    def __init__(self, name, min_mean, max_mean, lower_bound, round, min_std, max_std):
        self.name = name
        self.lower_bound = lower_bound
        self.round = round
        self.log_mean = UniformHyperparameter('log_mean', min=math.log(min_mean), max=math.log(max_mean))
        self.log_std = UniformHyperparameter('log_std', min=math.log(min_std), max=math.log(max_std))
    def __call__(self):
        return sample_meta(meta_trunc_norm_log_scaled, {"log_mean": self.log_mean, "log_std": self.log_std}, lower_bound=self.lower_bound, do_round=self.round)
    
class MetaTruncNormHyperparameter:
    def __init__(self, name, min_mean, max_mean, lower_bound, round, min_std, max_std):
        self.name = name
        self.lower_bound = lower_bound
        self.round = round
        self.mean = UniformHyperparameter('mean', min=min_mean, max=max_mean)
        self.std = UniformHyperparameter('std', min=min_std, max=max_std)

    def __call__(self):
        return sample_meta(make_trunc_norm, {"mean": self.log_mean, "std": self.std}, lower_bound=self.lower_bound, do_round=self.round)
    
class MetaChoiceHyperparameter:
    def __init__(self, name, choice_values):
        self.name = name
        self.choice_values = choice_values
        self.choices = {f"choice_{i}_weight": UniformHyperparameter(f"choice_{i}_weight", min=-3.0, max=5.0) for i in range(1, len(choice_values))}
    def __call__(self):
        return sample_meta(make_choice, self.choices, choice_values=self.choice_values)
    
class MetaChoiceMixedHyperparameter:
    def __init__(self, name, choice_values):
        self.name = name
        self.choice_values = choice_values
        self.choices = {f"choice_{i}_weight": UniformHyperparameter(f"choice_{i}_weight", min=-5.0, max=6.0) for i in range(1, len(choice_values))}
    def __call__(self):
        return sample_meta(make_choice_mixed, self.choices, choice_values=self.choice_values)


def parse_distribution(name, distribution, min=None, max=None, scale=None, lower_bound=None, round=None, min_mean=None, max_mean=None, min_std=0.01, max_std=1.0, choice_values=None,
                       max_alpha=None, max_scale=None):
    if distribution == "meta_beta":
        return MetaBetaHyperparameter(name=name, min=min, max=max, scale=scale)

    if distribution == "meta_gamma":
        return MetaGammaHyperparameter(name=name, max_alpha=max_alpha, max_scale=max_scale, lower_bound=lower_bound, round=round)

    elif distribution == "meta_trunc_norm_log_scaled":
        return MetaTruncNormLogScaledHyperparameter(
            name=name, min_mean=min_mean, max_mean=max_mean, lower_bound=lower_bound, round=round,
            min_std=min_std, max_std=max_std)
        
    elif distribution == "meta_trunc_norm":
        return MetaTruncNormHyperparameter(
            name=name, min_mean=min_mean, max_mean=max_mean, lower_bound=lower_bound, round=round,
            min_std=min_std, max_std=max_std)

    elif distribution == "meta_choice":
        return MetaChoiceHyperparameter(name=name, choice_values=choice_values)
    
    elif distribution == "meta_choice_mixed":
        return MetaChoiceMixedHyperparameter(name=name, choice_values=choice_values)
    elif distribution == "uniform":
        return UniformHyperparameter(name=name, min=min, max=max)
    else:
        raise ValueError(f"Distribution {distribution} not supported.")


class DifferentiablePrior:
    def __init__(self, get_batch, hyperparameters, differentiable_hyperparameters, args):

        self.h = hyperparameters
        self.args = args
        self.get_batch = get_batch
        self.differentiable_hyperparameters = {hp: parse_distribution(name=hp, **differentiable_hyperparameters[hp]) for hp in differentiable_hyperparameters}

    def __call__(self):
        # Sample hyperparameters
        sampled_hyperparameters_passed = {hp: hp_sampler() for hp, hp_sampler in self.differentiable_hyperparameters.items()}
        hyperparameters = {**self.h, **sampled_hyperparameters_passed}
        x, y, y_ = self.get_batch(hyperparameters=hyperparameters, **self.args)

        return x, y, y_


class DifferentiableSamplerPrior:
    def __init__(self, base_prior, differentiable_hyperparameters):
        self.base_prior = base_prior
        self.differentiable_hyperparameters = differentiable_hyperparameters

    def get_batch(self, batch_size, n_samples, num_features, device=default_device,
                  hyperparameters=None, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'batch_size': batch_size, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
            x, y, y_ = DifferentiablePrior(self.base_prior.get_batch, hyperparameters, self.differentiable_hyperparameters, args)()
            x, y, y_ = x.detach(), y.detach(), y_.detach()
        return x, y, y_, None
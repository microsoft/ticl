import math
import inspect

import torch


from tabpfn.utils import default_device

from .utils import (beta_sampler_f, gamma_sampler_f, uniform_int_sampler_f,
                    trunc_norm_sampler_f, uniform_sampler_f, log_uniform_sampler_f)

class HyperParameter:
    def __repr__(self):
        init_signature = inspect.signature(self.__init__)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return f"{self.__class__.__name__}({', '.join([f'{p.name}={getattr(self, p.name)}' for p in parameters])})"


class make_gamma(HyperParameter):
    def __init__(self, alpha, scale, do_round, lower_bound):
        self.alpha = alpha
        self.scale = scale
        self.do_round = do_round
        self.lower_bound = lower_bound
    def __call__(self):
        if self.do_round:
            return self.lower_bound + round(gamma_sampler_f(math.exp(self.alpha), self.scale / math.exp(self.alpha))())
        else:
            return self.lower_bound + gamma_sampler_f(math.exp(self.alpha), self.scale / math.exp(self.alpha))()


def meta_trunc_norm_log_scaled(log_mean, log_std, do_round, lower_bound):
    dist = trunc_norm_sampler_f(math.exp(log_mean), math.exp(log_mean)*math.exp(log_std))
    if do_round:
        return lambda: lower_bound + round(dist())
    else:    
        return lambda: lower_bound + dist()


def make_trunc_norm(mean, std, do_round, lower_bound):
    dist = trunc_norm_sampler_f(mean, std)
    if do_round:
        return lambda: lower_bound + round(dist())
    else:
        return lambda: lower_bound + dist()


class make_choice_mixed:
    def __init__(self, *, choice_values, choices):
        self.choice_values = choice_values
        self.choices = choices
        self.weights = torch.softmax(torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float), 0)  # create a tensor of weights

    def __call__(self):
        return self.sample
    
    def sample(self):
        s = torch.multinomial(self.weights, 1, replacement=True).numpy()[0]
        return self.choice_values[s]()
    
    def __repr__(self) -> str:
        return f'make_choice_mixed({self.choice_values}, {self.choices})'
    
    

def make_choice(*, choice_values, choices):
    choices = torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float)
    weights = torch.softmax(choices, 0)  # create a tensor of weights
    sample = torch.multinomial(weights, 1, replacement=True).numpy()[0]
    return choice_values[sample]
            


class UniformHyperparameter(HyperParameter):
    def __init__(self, name, min, max):
        self.min = min
        self.max = max
        self.name = name
    def __call__(self):
        return uniform_sampler_f(self.min, self.max)()
    

class LogUniformHyperparameter(HyperParameter):
    def __init__(self, name, min, max):
        self.min = min
        self.max = max
        self.name = name
    def __call__(self):
        return log_uniform_sampler_f(self.min, self.max)()
    
class UniformIntHyperparameter(HyperParameter):
    def __init__(self, name, min, max):
        self.min = min
        self.max = max
        self.name = name
    def __call__(self):
        return uniform_int_sampler_f(self.min, self.max)()


class MetaBetaHyperparameter(HyperParameter):
    def __init__(self, name, min, max, scale):
        self.scale = scale
        self.name = name
        self.min = min
        self.max = max
        self.b = UniformHyperparameter('b', min=min, max=max)
        self.k = UniformHyperparameter('k', min=min, max=max)

    def __call__(self):
        return beta_sampler_f(a=self.b(), b=self.k(), scale=self.scale)

class MetaGammaHyperparameter(HyperParameter):
    def __init__(self, name, max_alpha, max_scale, lower_bound, round):
        self.name = name
        self.max_alpha = max_alpha
        self.max_scale = max_scale
        self.alpha = UniformHyperparameter('alpha', min=0, max=math.log(max_alpha))  # Surprise Logarithm!
        self.scale = UniformHyperparameter('scale', min=0, max=max_scale)
        self.lower_bound = lower_bound
        self.round = round

    def __call__(self):
        return make_gamma(alpha=self.alpha(), scale=self.scale(), lower_bound=self.lower_bound, do_round=self.round)
    
class MetaTruncNormLogScaledHyperparameter(HyperParameter):
    def __init__(self, name, min_mean, max_mean, lower_bound, round, min_std, max_std):
        self.name = name
        self.lower_bound = lower_bound
        self.round = round
        self.min_mean = min_mean
        self.max_mean = max_mean    
        self.min_std = min_std
        self.max_std = max_std
        self.log_mean = UniformHyperparameter('log_mean', min=math.log(min_mean), max=math.log(max_mean))
        self.log_std = UniformHyperparameter('log_std', min=math.log(min_std), max=math.log(max_std))

    def __call__(self):
        return meta_trunc_norm_log_scaled(log_mean=self.log_mean(), log_std=self.log_std(), lower_bound=self.lower_bound, do_round=self.round)
    
class MetaTruncNormHyperparameter(HyperParameter):
    def __init__(self, name, min_mean, max_mean, lower_bound, round, min_std, max_std):
        self.name = name
        self.lower_bound = lower_bound
        self.round = round
        self.min_mean = min_mean
        self.max_mean = max_mean    
        self.min_std = min_std
        self.max_std = max_std
        self.mean = UniformHyperparameter('mean', min=min_mean, max=max_mean)
        self.std = UniformHyperparameter('std', min=min_std, max=max_std)

    def __call__(self):
        return make_trunc_norm(mean=self.log_mean(), std=self.std(), lower_bound=self.lower_bound, do_round=self.round)
    
class MetaChoiceHyperparameter(HyperParameter):
    def __init__(self, name, choice_values):
        self.name = name
        self.choice_values = choice_values
        self.choices = {f"choice_{i}_weight": UniformHyperparameter(f"choice_{i}_weight", min=-3.0, max=5.0) for i in range(1, len(choice_values))}
    def __call__(self):
        return make_choice(choices={choice: val() for choice, val in self.choices.items()}, choice_values=self.choice_values)

class MetaChoiceMixedHyperparameter(HyperParameter):
    def __init__(self, name, choice_values):
        self.name = name
        self.choice_values = choice_values
        self.choices = {f"choice_{i}_weight": UniformHyperparameter(f"choice_{i}_weight", min=-5.0, max=6.0) for i in range(1, len(choice_values))}
    def __call__(self):
        return make_choice_mixed(choices={choice: val() for choice, val in self.choices.items()}, choice_values=self.choice_values)


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
    elif distribution == "log_uniform":
        return LogUniformHyperparameter(name=name, min=min, max=max)
    elif distribution == "uniform_int":
        return UniformIntHyperparameter(name=name, min=min, max=max)
    else:
        raise ValueError(f"Distribution {distribution} not supported.")
    
def parse_distributions(hyperparameters):
    # parse any distributions in the hyperparameters that are represented as dicts
    new_hypers = {}
    for name, dist in hyperparameters.items():
        if isinstance(dist, dict) and "distribution" in dist:
            new_hypers[name] = parse_distribution(name=name, **dist)
        else:
            new_hypers[name] = dist
    return new_hypers


def sample_distributions(hyperparameters):
    # sample any distributions in the hyperparameters that are represented as functions
    new_hypers = {}
    for name, dist in hyperparameters.items():
        if isinstance(dist, HyperParameter):
            dist = dist()
        if callable(dist):
            dist = dist()
        new_hypers[name] = dist
    return new_hypers

class SamplerPrior:
    def __init__(self, base_prior, differentiable_hyperparameters, heterogeneous_batches=False):
        self.base_prior = base_prior
        self.heterogeneous_batches = heterogeneous_batches
        self.hyper_dists = {hp: parse_distribution(name=hp, **dist) for hp, dist in differentiable_hyperparameters.items()}

    def get_batch(self, batch_size, n_samples, num_features, device=default_device,
                  hyperparameters=None, epoch=None, single_eval_pos=None):
        args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'batch_size': batch_size, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
        with torch.no_grad():
            if self.heterogeneous_batches:
                args['batch_size'] = 1
                xs, ys, ys_ = [], [], []
                for i in range(0, batch_size):
                    sampled_hypers = {hp: dist() for hp, dist in self.hyper_dists.items()}
                    combined_hypers = {**hyperparameters, **sampled_hypers}
                    x, y, y_ = self.base_prior.get_batch(hyperparameters=combined_hypers, **args)
                    xs.append(x)
                    ys.append(y)
                    ys_.append(y_)
                    x, y, y_ = torch.cat(xs, 1), torch.cat(ys, 1), torch.cat(ys_, 1)
            else:
                sampled_hypers = {hp: dist() for hp, dist in self.hyper_dists.items()}
                combined_hypers = {**hyperparameters, **sampled_hypers}
                x, y, y_ = self.base_prior.get_batch(hyperparameters=combined_hypers, **args)
        x, y, y_ = x.detach(), y.detach(), y_.detach()
        return x, y, y_, sampled_hypers
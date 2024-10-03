import math
import inspect

import torch
import numpy as np
from scipy import stats


def safe_randint(low, high):
    if high <= low:
        return low
    return np.random.randint(low, high)


def trunc_norm_sampler_f(mu, sigma):
    dist = stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma)

    def sampler():
        return dist.rvs(1)[0]
    return sampler


class beta_sampler_f:
    def __init__(self, a, b, scale=1):
        self.a = a
        self.b = b
        self.scale = scale

    def __call__(self):
        return np.random.beta(self.a, self.b) * self.scale

    def __repr__(self) -> str:
        return f'beta_sampler_f({self.a},{self.b},{self.scale})'


def gamma_sampler_f(a, b): return lambda: np.random.gamma(a, b)
def uniform_sampler_f(a, b): return lambda: np.random.uniform(a, b)


class uniform_int_sampler_f:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f'uniform_int_sampler_f({self.a},{self.b})'

    def __call__(self):
        return round(np.random.uniform(self.a, self.b))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, uniform_int_sampler_f) and self.a == o.a and self.b == o.b


def log_uniform_sampler_f(a, b): return lambda: np.exp(np.random.uniform(np.log(a), np.log(b)))


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name='bounded_zipf', values=(x, weights)).rvs(1)


def scaled_beta_sampler_f(a, b, scale, minimum): return lambda: minimum + round(beta_sampler_f(a, b)() * (scale - minimum))


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
    for name, dist in sorted(hyperparameters.items(), key=lambda x: x[0]):
        if isinstance(dist, HyperParameter):
            dist = dist()
        if callable(dist) and not isinstance(dist, torch.nn.Module) and not isinstance(dist, type):
            dist = dist()
        new_hypers[name] = dist
    return new_hypers

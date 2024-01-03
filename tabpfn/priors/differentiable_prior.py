import math

import torch
from torch import nn

from tabpfn.utils import default_device

from .utils import (beta_sampler_f, gamma_sampler_f,
                    trunc_norm_sampler_f, uniform_int_sampler_f, uniform_sampler_f)


def unpack_dict_of_tuples(d):
    # Returns list of dicts where each dict i contains values of tuple position i
    # {'a': (1,2), 'b': (3,4)} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    return [dict(zip(d.keys(), v)) for v in list(zip(*list(d.values())))]

def get_sampler(distribution, min, max, sample):
    if distribution == "uniform":
        if sample is None:
            return uniform_sampler_f(min, max), min, max, (max+min) / 2, math.sqrt(1/12*(max-min)*(max-min))
        else:
            return lambda: sample, min, max, None, None
    elif distribution == "uniform_int":
        return uniform_int_sampler_f(min, max), min, max, (max+min) / 2, math.sqrt(1/12*(max-min)*(max-min))


def sample_meta(f, hparams, **kwargs):
    def sampler():
        passed = {hp: hparams[hp]() for hp in hparams}
        meta_passed = f(**passed, **kwargs)
        return meta_passed
    return sampler


def make_beta(b, k, scale):
    return lambda b=b, k=k: scale * beta_sampler_f(b, k)()


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

class DifferentiableHyperparameter(nn.Module):
    # We can sample this and get a hyperparameter value and a normalized hyperparameter indicator
    def __init__(self, distribution, embedding_dim, device, **args):
        super(DifferentiableHyperparameter, self).__init__()

        self.distribution = distribution
        self.embedding_dim = embedding_dim
        self.device = device
        for key in args:
            setattr(self, key, args[key])

        if self.distribution.startswith("meta"):
            self.hparams = {}
            args_passed = {'device': device, 'embedding_dim': embedding_dim}
            if self.distribution == "meta_beta":
                # Truncated normal where std and mean are drawn randomly logarithmically scaled
                if hasattr(self, 'b') and hasattr(self, 'k'):
                    self.hparams = {'b': lambda: (None, self.b), 'k': lambda: (None, self.k)}
                else:
                    self.hparams = {"b": DifferentiableHyperparameter(distribution="uniform", min=self.min, max=self.max, **args_passed),
                                    "k": DifferentiableHyperparameter(distribution="uniform", min=self.min, max=self.max, **args_passed)}

                self.sampler = sample_meta(make_beta, self.hparams, scale=self.scale)
            if self.distribution == "meta_gamma":
                # Truncated normal where std and mean are drawn randomly logarithmically scaled
                if hasattr(self, 'alpha') and hasattr(self, 'scale'):
                    self.hparams = {'alpha': lambda: (None, self.alpha), 'scale': lambda: (None, self.scale)}
                else:
                    self.hparams = {"alpha": DifferentiableHyperparameter(distribution="uniform", min=0.0, max=math.log(
                        self.max_alpha), **args_passed), "scale": DifferentiableHyperparameter(distribution="uniform", min=0.0, max=self.max_scale, **args_passed)}

                self.sampler = sample_meta(make_gamma, self.hparams, do_round=self.round, lower_bound=self.lower_bound)
            elif self.distribution == "meta_trunc_norm_log_scaled":
                # these choices are copied down below, don't change these without changing `replace_differentiable_distributions`
                self.min_std = self.min_std if hasattr(self, 'min_std') else 0.01
                self.max_std = self.max_std if hasattr(self, 'max_std') else 1.0
                # Truncated normal where std and mean are drawn randomly logarithmically scaled
                if not hasattr(self, 'log_mean'):
                    self.hparams = {"log_mean": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_mean), max=math.log(
                        self.max_mean), **args_passed), "log_std": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_std), max=math.log(self.max_std), **args_passed)}
                else:
                    self.hparams = {'log_mean': lambda: (None, self.log_mean), 'log_std': lambda: (None, self.log_std)}

                self.sampler = sample_meta(meta_trunc_norm_log_scaled, self.hparams, do_round=self.round, lower_bound=self.lower_bound)
            elif self.distribution == "meta_trunc_norm":
                self.min_std = self.min_std if hasattr(self, 'min_std') else 0.01
                self.max_std = self.max_std if hasattr(self, 'max_std') else 1.0
                self.hparams = {"mean": DifferentiableHyperparameter(distribution="uniform", min=self.min_mean, max=self.max_mean, **args_passed),
                                "std": DifferentiableHyperparameter(distribution="uniform", min=self.min_std, max=self.max_std, **args_passed)}
                self.sampler = sample_meta(make_trunc_norm, self.hparams, do_round=self.round, lower_bound=self.lower_bound)

            elif self.distribution == "meta_choice":
                self.hparams = {f"choice_{i}_weight": DifferentiableHyperparameter(
                        distribution="uniform", min=-3.0, max=5.0, **args_passed) for i in range(1, len(self.choice_values))}

                def make_choice(**choices):
                    choices = torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float)
                    weights = torch.softmax(choices, 0)  # create a tensor of weights
                    sample = torch.multinomial(weights, 1, replacement=True).numpy()[0]
                    return self.choice_values[sample]

                self.sampler = sample_meta(make_choice, self.hparams)

            elif self.distribution == "meta_choice_mixed":
                self.hparams = {f"choice_{i}_weight": DifferentiableHyperparameter(
                        distribution="uniform", min=-5.0, max=6.0, **args_passed) for i in range(1, len(self.choice_values))}

                def make_choice_mixed(**choices):
                    weights = torch.softmax(torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float), 0)  # create a tensor of weights

                    def sample():
                        s = torch.multinomial(weights, 1, replacement=True).numpy()[0]
                        return self.choice_values[s]()
                    return lambda: sample

                self.sampler = sample_meta(make_choice_mixed, self.hparams)
        else:
            self.sampler_f, self.sampler_min, self.sampler_max, self.sampler_mean, self.sampler_std = get_sampler(self.distribution, self.min, self.max, getattr(self, 'sample', None))
            self.sampler = self.sampler_f


    def forward(self):
        s_passed = self.sampler()
        return s_passed


class DifferentiableHyperparameterList(nn.Module):
    def __init__(self, hyperparameters, embedding_dim, device):
        super().__init__()

        self.device = device
        hyperparameters = {k: v for (k, v) in hyperparameters.items() if v}
        self.hyperparameters = nn.ModuleDict({hp: DifferentiableHyperparameter(embedding_dim=embedding_dim, name=hp,
                                             device=device, **hyperparameters[hp]) for hp in hyperparameters})

    def get_hyperparameter_info(self):
        sampled_hyperparameters_f, sampled_hyperparameters_keys = [], []

        def append_hp(hp_key, hp_val):
            sampled_hyperparameters_keys.append(hp_key)
            # Function remaps hyperparameters from [-1, 1] range to true value
            _, _, s_mean, s_std = hp_val.sampler_min, hp_val.sampler_max, hp_val.sampler_mean, hp_val.sampler_std
            sampled_hyperparameters_f.append((lambda x: (x-s_mean)/s_std, lambda y: (y * s_std)+s_mean))

        for hp in self.hyperparameters:
            hp_val = self.hyperparameters[hp]
            if hasattr(hp_val, 'hparams'):
                for hp_ in hp_val.hparams:
                    append_hp(f'{hp}_{hp_}', hp_val.hparams[hp_])
            else:
                append_hp(hp, hp_val)

        return sampled_hyperparameters_keys, sampled_hyperparameters_f

    def sample_parameter_object(self):
        s_passed = {hp: hp_sampler() for hp, hp_sampler in self.hyperparameters.items()}
        return s_passed


class DifferentiablePrior(torch.nn.Module):
    def __init__(self, get_batch, hyperparameters, differentiable_hyperparameters, args):
        super(DifferentiablePrior, self).__init__()

        self.h = hyperparameters
        self.args = args
        self.get_batch = get_batch
        self.differentiable_hyperparameters = DifferentiableHyperparameterList(
            differentiable_hyperparameters, embedding_dim=self.h['emsize'], device=self.args['device'])

    def forward(self):
        # Sample hyperparameters
        sampled_hyperparameters_passed = self.differentiable_hyperparameters.sample_parameter_object()
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
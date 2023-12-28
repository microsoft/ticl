import math
import random

import numpy as np
import scipy.stats as stats
import torch
from torch import nn

from tabpfn.utils import set_locals_in_self

from torch.utils.data import DataLoader


def eval_simple_dist(dist_dict):
    if not isinstance(dist_dict, dict):
        return dist_dict
    if dist_dict['distribution'] == "uniform":
        return np.random.uniform(dist_dict['min'], dist_dict['max'])
    raise ValueError("Distribution not supported")


class PriorDataLoader(DataLoader):
    def __init__(self, prior, num_steps, batch_size, eval_pos_n_samples_sampler, device, num_features, hyperparameters,
                    batch_size_per_prior_sample):
        self.prior = prior
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eval_pos_n_samples_sampler = eval_pos_n_samples_sampler
        self.device = device
        self.num_features = num_features
        self.hyperparameters = hyperparameters
        self.batch_size_per_prior_sample = batch_size_per_prior_sample
        self.epoch_count = 0

    def gbm(self, epoch=None):
        single_eval_pos, n_samples = self.eval_pos_n_samples_sampler()
        batch = self.prior.get_batch(batch_size=self.batch_size, n_samples=n_samples, num_features=self.num_features, device=self.device,
                                     hyperparameters=self.hyperparameters, batch_size_per_prior_sample=self.batch_size_per_prior_sample, epoch=epoch,
                                     single_eval_pos=single_eval_pos)
        x, y, target_y, style = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
        return (style, x, y), target_y, single_eval_pos
    
    def __len__(self):
        return self.num_steps

    def get_test_batch(self):  # does not increase epoch_count
        return self.gbm(epoch=self.epoch_count)

    def __iter__(self):
        self.epoch_count += 1
        return iter(self.gbm(epoch=self.epoch_count - 1) for _ in range(self.num_steps))

def trunc_norm_sampler_f(mu, sigma): return lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
def beta_sampler_f(a, b): return lambda: np.random.beta(a, b)
def gamma_sampler_f(a, b): return lambda: np.random.gamma(a, b)
def uniform_sampler_f(a, b): return lambda: np.random.uniform(a, b)
def uniform_int_sampler_f(a, b): return lambda: round(np.random.uniform(a, b))


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name='bounded_zipf', values=(x, weights)).rvs(1)


def scaled_beta_sampler_f(a, b, scale, minimum): return lambda: minimum + round(beta_sampler_f(a, b)() * (scale - minimum))


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(n_samples)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(n_samples, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(n_samples, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7, keep_activation_size=False, num_classes_sampler=zipf_sampler_f(0.8, 1, 10)):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x

import math
import random

import numpy as np
import scipy.stats as stats
import torch
from torch import nn

from tabpfn.utils import set_locals_in_self

from torch.utils.data import DataLoader


def get_batch_to_dataloader(get_batch_method_):
    class DL(DataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps, **get_batch_kwargs):
            set_locals_in_self(locals())

            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features = get_batch_kwargs.get('num_features') or self.num_features
            self.epoch_count = 0
            # print('DataLoader.__dict__', self.__dict__)

        @staticmethod
        def gbm(*args, eval_pos_seq_len_sampler, **kwargs):
            kwargs['single_eval_pos'], kwargs['seq_len'] = eval_pos_seq_len_sampler()
            # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
            # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
            if 'dynamic_batch_size' in kwargs and kwargs['dynamic_batch_size'] > 0 and kwargs['dynamic_batch_size']:
                kwargs['batch_size'] = kwargs['batch_size'] * \
                    math.floor(math.pow(kwargs['seq_len_maximum'], kwargs['dynamic_batch_size']) / math.pow(kwargs['seq_len'], kwargs['dynamic_batch_size']))
            batch = get_batch_method_(*args, **kwargs)
            x, y, target_y, style = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
            return (style, x, y), target_y, kwargs['single_eval_pos']

        def __len__(self):
            return self.num_steps

        def get_test_batch(self):  # does not increase epoch_count
            return self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count)

        def __iter__(self):
            self.epoch_count += 1
            return iter(self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1) for _ in range(self.num_steps))

    return DL

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
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

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

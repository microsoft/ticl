import numpy as np
import torch
from ticl.utils import default_device, normalize_data
from ticl.distributions import safe_randint


def sample_boolean_data_enumerate(hyperparameters, n_samples, num_features):
    # unused, might be better? unclear.
    rank = np.random.randint(1, min(10, num_features))
    grid = torch.meshgrid([torch.tensor([-1, 1])] * num_features)
    inputs = torch.stack(grid, dim=-1).view(-1, num_features)
    outputs = torch.zeros(2**num_features, dtype=bool)

    while 3 * torch.sum(outputs) < len(inputs):
        selected_bits = torch.multinomial(torch.ones(num_features), rank, replacement=False)
        signs = torch.randint(2, (rank,))*2-1
        outputs = outputs + ((signs * inputs[:, selected_bits]) == 1).all(dim=1)
    return (inputs + 1) / 2, outputs


class BooleanConjunctionPrior:
    # This is a class mostly for debugging purposes
    # the object stores the sampled hyperparameters
    def __init__(self, hyperparameters=None, debug=False):
        if hyperparameters is None:
            hyperparameters = {}
        self.debug = debug
        self.max_rank = hyperparameters.get("max_rank", 10)
        self.verbose = hyperparameters.get("verbose", False)
        self.max_fraction_uninformative = hyperparameters.get("max_fraction_uninformative", 0.5)
        self.p_uninformative = hyperparameters.get("p_uninformative", 0.5)

    def sample(self, n_samples, num_features, device):
        # num_features is always 100, i.e. the number of inputs of the transformer model
        # num_features_active is the number of synthetic datasets features
        # num_features_important is the number of features that actually determine the output
        num_features_important = safe_randint(1, num_features)
        if np.random.random() < self.p_uninformative:
            num_features_active = num_features_important + \
                min(safe_randint(1, int(self.max_fraction_uninformative * num_features_important)), num_features - num_features_important)
        else:
            num_features_active = num_features_important
        rank = safe_randint(1, min(self.max_rank, num_features_important))
        num_terms_max = int(np.exp((rank - 1) / 1.5))
        inputs = 2 * torch.randint(0, 2, (n_samples, num_features_active), device=device) - 1
        important_indices = torch.randperm(num_features_active)[:num_features_important]
        inputs_important = inputs[:, important_indices]
        selected_bits = torch.multinomial(torch.ones(num_features_important, device=device) / num_features_important,
                                          rank * num_terms_max, replacement=True).reshape(rank, num_terms_max)
        signs = torch.randint(2, (rank, num_terms_max), device=device) * 2 - 1
        outputs = ((signs * inputs_important[:, selected_bits]) == 1).all(dim=1).any(dim=1)
        sample_params = {'num_terms': num_terms_max, 'rank': rank, 'important_indices': important_indices,
                         'num_features_active': num_features_active, 'num_features_important': num_features_important, 'num_features': num_features}
        if self.debug:
            sample_params['features_in_terms'] = selected_bits.unique()
        return inputs, outputs, sample_params

    def normalize_and_pad(self, x, y, num_features, device):
        x = torch.cat([x, torch.zeros(x.shape[0], num_features - x.shape[1], device=device)], dim=1)
        xs, ys = ((x + 1) / 2).unsqueeze(1), y.int().unsqueeze(1).unsqueeze(2)
        xs = normalize_data(xs)
        return xs, ys

    def __call__(self, n_samples, num_features, device):
        x, y, sample_params, = self.sample(n_samples, num_features, device)
        return *self.normalize_and_pad(x, y, num_features, device), sample_params

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None, **kwargs):
        assert num_outputs == 1
        sample = [self(n_samples=n_samples, num_features=num_features, device=device) for _ in range(0, batch_size)]
        x, y, _ = zip(*sample)
        y = torch.cat(y, 1).detach().squeeze(2)
        x = torch.cat(x, 1).detach()

        return x, y, y, {}

import numpy as np
import torch
from tabpfn.utils import default_device
from tabpfn.priors.utils import get_batch_to_dataloader


def sample_boolean_data_enumerate(hyperparameters, seq_len, num_features):
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


def sample_boolean_data(hyperparameters, seq_len, num_features, device):
    num_features_active = np.random.randint(1, num_features) if num_features > 1 else 1
    max_rank = hyperparameters.get("max_rank", 10)
    rank = np.random.randint(1, min(max_rank, num_features_active)) if min(max_rank, num_features_active) > 1 else 1
    n_samples = seq_len
    inputs = 2 * torch.randint(0, 2, (n_samples, num_features_active), device=device) - 1
    outputs = torch.zeros(n_samples, dtype=bool, device=device)

    while 3 * torch.sum(outputs) < len(inputs):
        selected_bits = torch.multinomial(torch.ones(num_features_active, device=device), rank, replacement=False)
        signs = torch.randint(2, (rank,), device=device) * 2 - 1
        outputs = outputs + ((signs * inputs[:, selected_bits]) == 1).all(dim=1)
    inputs = torch.cat([inputs, torch.zeros(n_samples, num_features - num_features_active, device=device)], dim=1)
    return ((inputs + 1) / 2).unsqueeze(1), outputs.int().unsqueeze(1).unsqueeze(2)


def get_batch(batch_size, seq_len, num_features, hyperparameters, device=default_device, num_outputs=1, sampling='normal', epoch=None, **kwargs):
    assert num_outputs == 1
    sample = [sample_boolean_data(hyperparameters, seq_len=seq_len, num_features=num_features, device=device) for _ in range(0, batch_size)]
    x, y = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    x = torch.cat(x, 1).detach()

    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)

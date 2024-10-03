import torch
import torch.nn as nn
import math

from ticl.utils import normalize_data


def get_fourier_features(x, n_features):
    div_term = torch.exp(torch.arange(0, n_features, 2, device=x.device) * (-math.log(10000.0) / n_features))
    x = x.unsqueeze(-1)
    return torch.cat([x, torch.sin(x * div_term), torch.cos(x * div_term)], -1)


class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat([torch.nan_to_num(x, nan=0.0), normalize_data(torch.isnan(x) * -1
                                                                        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                                                                        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                                                                        )], -1)
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)


class BinEmbeddingEncoder(nn.Module):
    def __init__(self, num_features, emsize, n_bins, rank, nonlinear=True, decoder_activation='relu'):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.n_bins = n_bins
        self.rank = rank
        self.nonlinear = nonlinear
        self.embedding = nn.Parameter(torch.randn(n_bins, rank))
        self.bias = nn.Parameter(torch.randn(1, 1, num_features, rank))
        self.weights = nn.Parameter(torch.randn(num_features, rank, emsize))
        self.decoder_activation = decoder_activation

    def forward(self, x):
        # n samples, b batch, k feature, d bins, r rank
        embedded = torch.einsum('nbkd,dr->nbkr', x, self.embedding)
        if self.nonlinear:
            embedded = embedded + self.bias
            if self.decoder_activation == 'relu':
                embedded = torch.nn.functional.relu(embedded)
            elif self.decoder_activation == 'gelu':
                embedded = torch.nn.functional.gelu(embedded)
            else:
                raise ValueError(f"decoder_activation {self.decoder_activation} not supported")
        # n samples, b batch, k feature, r rank, e embedding dim in transformer
        out = torch.einsum('nbkr,kre->nbe', embedded, self.weights)
        return out


class OneHotAndLinear(nn.Linear):
    def __init__(self, num_classes, emsize):
        super().__init__(num_classes, emsize)
        self.num_classes = num_classes
        self.emsize = emsize

    def forward(self, x, inference: bool = False):
        if (x == -100).any():
            pass
        y = x.squeeze().long()
        if inference:
            assert len(y.shape) == 1, "Inference mode only supports batch size 1"
            y = y.unsqueeze(0)
        mask = y == -100
        y[mask] = 0
        out = torch.nn.functional.one_hot(y, self.num_classes).float()

        if out.ndim == 3:
            out[:, :, 0][mask] = 0
        else:
            out[:, 0][mask] = 0
            out = out.unsqueeze(1)
        return super().forward(out)

# taken from """Revisiting Deep Learning Models for Tabular Data."""
# https://github.com/yandex-research/rtdl-revisiting-models/
# version 0.0.2


from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import torch.nn as nn
from torch import Tensor

from ticl.evaluation.baselines.torch_mlp import TorchModelTrainer


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class ResNet(nn.Module):
    """The ResNet model from Section 3.2 in the paper."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        d_hidden: Optional[int] = None,
        d_hidden_multiplier: Optional[float],
        dropout1: float,
        dropout2: float,
    ) -> None:
        """
        Args:
            d_in: the input size.
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width (i.e. its input and output size).
            d_hidden: the block's hidden width.
            d_hidden_multipler: the alternative way to set `d_hidden` as
                `int(d_block * d_hidden_multipler)`.
            dropout1: the hidden dropout rate.
            dropout2: the residual dropout rate.
        """
        if n_blocks <= 0:
            raise ValueError(f'n_blocks must be positive, however: {n_blocks=}')
        if d_hidden is None:
            if d_hidden_multiplier is None:
                raise ValueError(
                    'If d_hidden is None, then d_hidden_multiplier must not be None'
                )
            d_hidden = int(d_block * cast(float, d_hidden_multiplier))
        else:
            if d_hidden_multiplier is not None:
                raise ValueError(
                    'If d_hidden is None, then d_hidden_multiplier must be None'
                )

        super().__init__()
        self.input_projection = nn.Linear(d_in, d_block)
        self.blocks = nn.ModuleList(
            [
                _named_sequential(
                    ('normalization', nn.BatchNorm1d(d_block)),
                    ('linear1', nn.Linear(d_block, d_hidden)),
                    ('activation', nn.ReLU()),
                    ('dropout1', nn.Dropout(dropout1)),
                    ('linear2', nn.Linear(d_hidden, d_block)),
                    ('dropout2', nn.Dropout(dropout2)),
                )
                for _ in range(n_blocks)
            ]
        )
        self.output = (
            None
            if d_out is None
            else _named_sequential(
                ('normalization', nn.BatchNorm1d(d_block)),
                ('activation', nn.ReLU()),
                ('linear', nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)
        if self.output is not None:
            x = self.output(x)
        return x


class ResNetClassifier(TorchModelTrainer):
    def __init__(self, hidden_size=128, hidden_multiplier=2, n_epochs=10, learning_rate=1e-3, n_layers=2,
                 verbose=0, dropout_rate=0.0, device='cuda', weight_decay=0.01, batch_size=None, epoch_callback=None,
                 init_state=None,):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.hidden_multiplier = hidden_multiplier
        self.dropout_rate = dropout_rate
        super().__init__(n_epochs=n_epochs, learning_rate=learning_rate, verbose=verbose,
                         device=device, init_state=init_state, batch_size=batch_size,
                         epoch_callback=epoch_callback, weight_decay=weight_decay)

    def make_model(self, n_features, n_classes):
        return ResNet(d_in=n_features, d_out=n_classes, n_blocks=self.n_layers, d_block=self.hidden_size,
                      d_hidden_multiplier=1.0, dropout1=self.dropout_rate, dropout2=self.dropout_rate)
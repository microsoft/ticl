from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from tabpfn.models.layer import TransformerEncoderLayer
from tabpfn.utils import SeqBN


class TabPFN(nn.Module):
    def __init__(self, encoder_layer, *, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, y_encoder=None):
        super().__init__()
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor

        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.emsize = emsize
        self.encoder = encoder_layer
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking

        self.n_out = n_out
        self.nhid = nhid

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)

    def init_weights(self):
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        style_src = torch.tensor([], device=x_src.device)
        global_src = torch.tensor([], device=x_src.device)

        if src_mask is None:
            full_len = len(x_src) + len(style_src)
            if self.efficient_eval_masking:
                src_mask = single_eval_pos + len(style_src)
            else:
                src_mask = self.generate_D_q_matrix(full_len, len(x_src) - single_eval_pos).to(x_src.device)

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[single_eval_pos+len(style_src):]


class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

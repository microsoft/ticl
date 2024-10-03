from typing import Optional

import torch, wandb
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from ticl.models.layer import TransformerEncoderLayer
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear


class TabPFN(nn.Module):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, causal_mask=False, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, classification_task=True,
                 all_layers_same_init=False, efficient_eval_masking=True, y_encoder=None, tabpfn_zero_weights=False):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor

        self.causal_mask = causal_mask

        def encoder_layer_creator(): return TransformerEncoderLayer(
            emsize, 
            nhead, 
            nhid, 
            dropout, 
            activation=activation,
            pre_norm=pre_norm, 
            recompute_attn=recompute_attn,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if single_eval_pos is None: raise ValueError('single_eval_pos has to be given, instead of None.')

        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        
        # x_src: (num_samples, batch_size, d_model)
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        if self.efficient_eval_masking:
            src_mask = single_eval_pos
        else:
            raise NotImplementedError(f'efficient_eval_masking={self.efficient_eval_masking} is not implemented yet.')

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask, is_causal = self.causal_mask)
        # decoder is some position-wise operation
        output = self.decoder(output)
        return output[single_eval_pos:]

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

    def forward(
        self, 
        src: Tensor, 
        mask: Optional[Tensor] = None, 
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = False,
    ) -> Tensor:
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
            output = mod(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask, 
                is_causal=is_causal
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

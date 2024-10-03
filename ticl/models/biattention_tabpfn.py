import torch
import torch.nn as nn

from ticl.models.layer import BiAttentionEncoderLayer
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear, get_fourier_features


class BiAttentionTabPFN(nn.Module):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, efficient_eval_masking=True, classification_task=True,
                 all_layers_same_init=False,  y_encoder=None, tabpfn_zero_weights=False, input_embedding='linear'):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor

        def encoder_layer_creator(): return BiAttentionEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(nlayers)])
        self.emsize = emsize
        self.input_embedding = input_embedding
        if input_embedding == "linear":
            self.encoder = Linear(1, emsize, replace_nan_by_zero=True)
        elif input_embedding == "random":
            self.encoder = nn.LayerNorm(normalized_shape=[emsize])
        elif input_embedding == "fourier":
            self.encoder = Linear(emsize + 1, emsize, replace_nan_by_zero=True)
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for bilayer in self.layers:
                for layer in [bilayer.cross_feature_attention, bilayer.cross_sample_attention]:
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
                    attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                    for attn in attns:
                        nn.init.zeros_(attn.out_proj.weight)
                        nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        if self.input_embedding == "linear":
            x_src = self.encoder(x_src.unsqueeze(-1))
        elif self.input_embedding == "random":
            proj = torch.randn(x_src.shape[-2], x_src.shape[-1], self.emsize, device=x_src.device, dtype=x_src.dtype)
            x_src = x_src.unsqueeze(-1).nan_to_num(0) * proj
            x_src = self.encoder(x_src)
        elif self.input_embedding == "fourier":
            x_fourier = get_fourier_features(x_src, self.emsize)
            x_src = self.encoder(x_fourier)
        else:
            raise ValueError(f"input_embedding {self.input_embedding} not supported")
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        if src_mask is None:
            src_mask = single_eval_pos

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos].unsqueeze(-2)
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)
        if self.input_ln is not None:
            src = self.input_ln(src)
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        decoded_output = self.decoder(output.mean(axis=-2))
        return decoded_output[single_eval_pos:]

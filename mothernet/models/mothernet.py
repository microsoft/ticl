import torch
import torch.nn as nn
from torch.nn import TransformerEncoder

from mothernet.models.encoders import OneHotAndLinear
from mothernet.models.decoders import LinearModelDecoder, MLPModelDecoder
from mothernet.models.layer import TransformerEncoderLayer
from mothernet.models.tabpfn import TransformerEncoderDiffInit
from mothernet.utils import SeqBN


class MLPModelPredictor(nn.Module):
    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        _, x, y = src
        x_enc = self.encoder(x)
        y_enc = self.y_encoder(y.unsqueeze(-1) if len(y.shape) < len(x.shape) else y)
        enc_train = x_enc[:single_eval_pos] + y_enc[:single_eval_pos]
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat([self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0)
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError("class_tokens decoder type is only supported with OneHotAndLinear y_encoder")
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        output = self.inner_forward(enc_train)
        (b1, w1), *layers = self.decoder(output, y[:single_eval_pos])

        x_test_nona = torch.nan_to_num(x[single_eval_pos:], nan=0)
        h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

        if self.decoder.weight_embedding_rank is not None:
            h = torch.matmul(h, self.decoder.shared_weights[0])
        h = h + b1

        for i, (b, w) in enumerate(layers):
            h = torch.relu(h)
            h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
            if self.decoder.weight_embedding_rank is not None and i != len(layers) - 1:
                # last layer has no shared weights
                h = torch.matmul(h, self.decoder.shared_weights[i + 1])
            h = h + b

        if h.isnan().all():
            print("NAN")
            raise ValueError("NAN")
            import pdb
            pdb.set_trace()
        return h


class MotherNet(MLPModelPredictor):
    def __init__(self, encoder_layer, *, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_type="output_attention", predicted_hidden_layer_size=None,
                 decoder_embed_dim=2048,
                 decoder_hidden_layers=1, decoder_hidden_size=None, predicted_hidden_layers=1, weight_embedding_rank=None, y_encoder=None,
                 low_rank_weights=False):
        super().__init__()
        nhid = emsize * nhid_factor
        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.emsize = emsize
        self.encoder = encoder_layer
        self.y_encoder = y_encoder_layer
        self.decoder = LinearModelDecoder(emsize=emsize, hidden_size=nhid, n_out=n_out)
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = n_out
        self.nhid = nhid
        self.decoder_type = decoder_type
        decoder_hidden_size = decoder_hidden_size or nhid
        self.decoder = MLPModelDecoder(emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out, decoder_type=self.decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights)
        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))

        self.init_weights()

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

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)

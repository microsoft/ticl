import torch
import torch.nn as nn


from mothernet.models.layer import BiAttentionEncoderLayer
from mothernet.models.decoders import FactorizedAdditiveModelDecoder, AdditiveModelDecoder
from mothernet.models.encoders import BinEmbeddingEncoder, Linear, OneHotAndLinear

from mothernet.utils import SeqBN, get_init_method


class BiAttentionMotherNetAdditive(nn.Module):
    def __init__(self, *, n_features, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_embed_dim=2048, low_rank_weights=None, weight_embedding_rank=None,
                 decoder_hidden_layers=1, decoder_hidden_size=None, n_bins=64, input_bin_embedding=False,
                 bin_embedding_rank=16, output_rank=16, factorized_output=False, y_encoder=None,
                 predicted_hidden_layer_size=None, predicted_hidden_layers=None,
                 decoder_type=None, input_layer_norm=False, shape_attention=False, tabpfn_zero_weights=True, shape_attention_heads=1, n_shape_functions=32,
                 shape_init="constant"):
        super().__init__()
        nhid = emsize * nhid_factor
        self.y_encoder = y_encoder_layer
        self.low_rank_weights = low_rank_weights  # ignored for now
        self.weight_embedding_rank = weight_embedding_rank  # ignored for now

        def encoder_layer_creator(): return BiAttentionEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(nlayers)])
        self.emsize = emsize

        if input_bin_embedding == "linear":
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=False)
        elif input_bin_embedding in ["True", "nonlinear"] or isinstance(input_bin_embedding, bool) and input_bin_embedding:
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=True)
        elif input_bin_embedding in ["none", "False"] or isinstance(input_bin_embedding, bool) and not input_bin_embedding:
            self.encoder = Linear(num_features=n_bins, emsize=emsize, replace_nan_by_zero=True)
        else:
            raise ValueError(f"Unknown input_bin_embedding: {input_bin_embedding}")

        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_bins = n_bins
        self.n_out = n_out
        self.nhid = nhid
        self.input_bin_embedding = input_bin_embedding
        self.output_rank = output_rank
        self.factorized_output = factorized_output
        self.decoder_type = decoder_type
        self.input_layer_norm = input_layer_norm
        self.shape_attention = shape_attention
        self.tabpfn_zero_weights = tabpfn_zero_weights

        if factorized_output:
            self.decoder = FactorizedAdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                          embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                          decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, rank=output_rank,
                                                          shape_attention=shape_attention, shape_attention_heads=shape_attention_heads,
                                                          n_shape_functions=n_shape_functions, shape_init=shape_init, biattention=True)
        else:
            self.decoder = AdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                decoder_hidden_layers=decoder_hidden_layers, nhead=nhead)
            
        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))
        if self.input_layer_norm:
            self.input_norm = nn.LayerNorm(normalized_shape=(n_features, n_bins))
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

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        _, x_src_org, y_src_org = src
        X_onehot, _ = bin_data(x_src_org, n_bins=self.n_bins,
                               single_eval_pos=single_eval_pos)
        X_onehot = X_onehot.float()
        if self.input_layer_norm:
            X_onehot = self.input_norm(X_onehot)

        x_src = self.encoder(X_onehot)
        y_src = self.y_encoder(y_src_org.unsqueeze(-1) if len(y_src_org.shape) < len(x_src.shape) else y_src_org)

        enc_train = x_src[:single_eval_pos] + y_src[:single_eval_pos].unsqueeze(-2)

        # FIXME Refactor into a function to share with mothernet
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat([self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0)
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError("class_tokens decoder type is only supported with OneHotAndLinear y_encoder")
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        if self.input_ln is not None:
            enc_train = self.input_ln(enc_train)
        output = enc_train
        for mod in self.layers:
            output = mod(output, src_mask=single_eval_pos)

        weights, biases = self.decoder(output, y_src_org[:single_eval_pos])
        # n samples, b batch, k feature, d bins, o outputs
        h = torch.einsum("nbkd,bkdo->nbo", X_onehot[single_eval_pos:], weights)
        h = h + biases

        if h.isnan().all():
            print("NAN")
            import pdb
            pdb.set_trace()
        return h


def bin_data(data, n_bins, single_eval_pos=None):
    # data is samples x batch x features
    # FIXME treat NaN as separate bin
    data_nona = torch.nan_to_num(data, nan=0)
    quantiles = torch.arange(n_bins + 1, device=data.device) / n_bins
    if single_eval_pos is None:
        bin_edges = torch.quantile(data_nona, quantiles[1:-1], dim=0)
    else:
        bin_edges = torch.quantile(data_nona[:single_eval_pos], quantiles[1:-1], dim=0)
    zero_padding = (data_nona == 0).all(axis=0)
    # FIXME extra data copy
    bin_edges = bin_edges.transpose(0, -1).contiguous()
    data_nona = data_nona.transpose(0, -1).contiguous()
    X_binned = torch.searchsorted(bin_edges, data_nona)
    X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    # mask zero padding data
    X_onehot[:, zero_padding, :] = 0
    return X_onehot, bin_edges





class BiAttentionTabPFN(nn.Module):
    def __init__(self, encoder_layer, *, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, efficient_eval_masking=True,
                 all_layers_same_init=False,  y_encoder=None, tabpfn_zero_weights=False, input_embedding='linear'):
        super().__init__()
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor


        self.emsize = emsize
        self.input_embedding = input_embedding
        if input_embedding == "linear":
            self.encoder = encoder_layer
        elif input_embedding == "random":
            self.encoder = nn.LayerNorm(normalized_shape=[emsize])
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
            x_src = x_src.unsqueeze(-1) * proj
            x_src = self.encoder(x_src)

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
        output = self.decoder(output.mean(axis=-2))
        return output[single_eval_pos:]
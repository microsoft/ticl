import torch
import torch.nn as nn

from tabpfn.models.decoders import AdditiveModelDecoder, FactorizedAdditiveModelDecoder
from tabpfn.models.encoders import BinEmbeddingEncoder, Linear
from tabpfn.models.layer import TransformerEncoderLayer
from tabpfn.models.transformer import TransformerEncoderDiffInit
from tabpfn.utils import SeqBN, bool_mask_to_att_mask


class MotherNetAdditive(nn.Module):
    def __init__(self, *, n_features, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_embed_dim=2048, low_rank_weights=None, weight_embedding_rank=None,
                 decoder_two_hidden_layers=False, decoder_hidden_size=None, n_bins=64, input_bin_embedding=False,
                 bin_embedding_rank=16, output_rank=16, factorized_output=False, y_encoder=None,
                 predicted_hidden_layer_size=None, output_attention=None, special_token=None, no_double_embedding=None, predicted_hidden_layers=None):
        super().__init__()
        nhid = emsize *  nhid_factor
        self.y_encoder = y_encoder # unused for now, y_encoder_layer was passed
        self.low_rank_weights = low_rank_weights # ignored for now
        self.weight_embedding_rank = weight_embedding_rank # ignored for now
        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.emsize = emsize
        if input_bin_embedding:
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank)
        else:
            self.encoder = Linear(num_features=n_features*n_bins, emsize=emsize, replace_nan_by_zero=True)
        self.y_encoder = y_encoder_layer
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking
        self.n_bins = n_bins
        self.n_out = n_out
        self.nhid = nhid
        self.input_bin_embedding = input_bin_embedding
        self.output_rank = output_rank
        self.factorized_output = factorized_output

        if factorized_output:
            self.decoder = FactorizedAdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                          embed_dim=decoder_embed_dim,
                                                          decoder_two_hidden_layers=decoder_two_hidden_layers, nhead=nhead, rank=output_rank)
        else:
            self.decoder = AdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                embed_dim=decoder_embed_dim,
                                                decoder_two_hidden_layers=decoder_two_hidden_layers, nhead=nhead)

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        # ?!?!? FIXME THIS SEEMS WRONG
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global_att_tokens, n_samples, num_query_tokens):
        train_size = n_samples + num_global_att_tokens - num_query_tokens
        sz = n_samples + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:, train_size:].zero_()
        mask[:, train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(num_global_att_tokens, n_samples, num_query_tokens):
        trainset_size = n_samples - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global_att_tokens, n_samples, num_query_tokens):
        mask = torch.zeros(num_global_att_tokens, num_global_att_tokens+n_samples-num_query_tokens) == 0
        return bool_mask_to_att_mask(mask)

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

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        _, x_src_org, y_src = src
        X_onehot, _ = bin_data(x_src_org, n_bins=self.n_bins)
        X_onehot = X_onehot.float()
        if self.input_bin_embedding:
            X_onehot_flat = X_onehot
        else:
            # would be more elegant to do this in the actual encoder
            X_onehot_flat = X_onehot.reshape((*X_onehot.shape[:-2], -1))

        x_src = self.encoder(X_onehot_flat)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]

        output = self.transformer_encoder(train_x)
        weights, biases = self.decoder(output)
        # n samples, b batch, k feature, d bins, o outputs
        h = torch.einsum("nbkd,bkdo->nbo", X_onehot[single_eval_pos:], weights)
        h = h + biases

        if h.isnan().all():
            print("NAN")
            import pdb
            pdb.set_trace()
        return h


def bin_data(data, n_bins):
    # FIXME treat NaN as separate bin
    data_nona = torch.nan_to_num(data, nan=0)
    quantiles = torch.arange(n_bins + 1, device=data.device) / n_bins
    bin_edges = torch.quantile(data_nona, quantiles[1:-1], dim=0)
    zero_padding = (data_nona == 0).all(axis=0)
    # FIXME extra data copy
    bin_edges = bin_edges.transpose(0, -1).contiguous()
    data_nona = data_nona.transpose(0, -1).contiguous()
    X_binned = torch.searchsorted(bin_edges, data_nona)
    X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    # mask zero padding data
    X_onehot[:, zero_padding, :] = 0
    return X_onehot, bin_edges

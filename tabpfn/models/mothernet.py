import torch
import torch.nn as nn
from torch.nn import TransformerEncoder

from tabpfn.models.decoders import LinearModelDecoder, MLPModelDecoder
from tabpfn.models.layer import TransformerEncoderLayer
from tabpfn.models.transformer import TransformerEncoderDiffInit
from tabpfn.utils import SeqBN, bool_mask_to_att_mask


class MLPModelPredictor(nn.Module):
    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        _, x_src_org, y_src = src
        x_src = self.encoder(x_src_org)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        if self.special_token:
            train_x = torch.cat([self.token_embedding.repeat(1, train_x.shape[1], 1), train_x], 0)

        output = self.inner_forward(train_x)
        (b1, w1), *layers = self.decoder(output)

        if self.no_double_embedding:
            x_src_org_nona = torch.nan_to_num(x_src_org[single_eval_pos:], nan=0)
            h = (x_src_org_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)
        else:
            h = (x_src[single_eval_pos:].unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

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
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, output_attention=False, special_token=False, predicted_hidden_layer_size=None, decoder_embed_dim=2048,
                 decoder_two_hidden_layers=False, decoder_hidden_size=None, no_double_embedding=False, predicted_hidden_layers=1, weight_embedding_rank=None):
        super().__init__()
        self.model_type = 'Transformer'
        def encoder_layer_creator(): return TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = LinearModelDecoder(emsize=ninp, hidden_size=nhid, n_out=n_out)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = n_out
        self.nhid = nhid
        self.no_double_embedding = no_double_embedding
        self.output_attention = output_attention
        self.special_token = special_token
        decoder_hidden_size = decoder_hidden_size or nhid
        self.decoder = MLPModelDecoder(emsize=ninp, hidden_size=decoder_hidden_size, n_out=n_out, output_attention=self.output_attention,
                                       special_token=special_token, predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_two_hidden_layers=decoder_two_hidden_layers, no_double_embedding=no_double_embedding, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank)
        if special_token:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, ninp))

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

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)
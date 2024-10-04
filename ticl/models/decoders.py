import torch
from torch import nn
import numpy as np


class LinearModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_out=10, hidden_size=1024):
        super().__init__()
        self.emsize = emsize
        self.n_out = n_out
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(nn.Linear(emsize,  hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, (emsize + 1) * n_out))

    def forward(self, x):
        return self.mlp(x.mean(0)).reshape(-1, self.emsize + 1, self.n_out)


class AdditiveModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_features=100, n_bins=64, n_out=10, hidden_size=1024, predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_hidden_layers=1, nhead=4, weight_embedding_rank=None, decoder_type="output_attention", biattention=False, decoder_activation='relu',
                 shape_init=None, add_marginals=False):
        super().__init__()
        self.emsize = emsize
        self.n_features = n_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank
        self.biattention = biattention
        self.decoder_activation = decoder_activation
        self.decoder_type = decoder_type
        self.summary_layer = SummaryLayer(emsize=emsize, n_out=n_out, decoder_type=decoder_type, embed_dim=embed_dim, nhead=nhead)
        self.add_marginals = add_marginals
        
        if decoder_type in ["class_tokens", "class_average"]:
            self.num_output_layer_weights = n_bins if biattention else n_bins * n_features + 1
            mlp_in_size = emsize
        else:
            mlp_in_size = self.summary_layer.out_size
            self.num_output_layer_weights = n_out * n_bins if biattention else n_out * (n_bins * n_features + 1)
            if add_marginals:
                raise ValueError("add_marginals is not supported for non-class models")

        if add_marginals and not biattention:
            raise ValueError("Biattention=False and add_marginals are not supported together")

        if add_marginals:
            mlp_in_size = mlp_in_size + n_bins

        self.mlp = make_decoder_mlp(mlp_in_size, hidden_size, self.num_output_layer_weights, n_layers=decoder_hidden_layers, activation=decoder_activation)
        if shape_init == "zero":
            with torch.no_grad():
                self.mlp[2].weight.data.fill_(0)
                self.mlp[2].bias.data.fill_(0)

    def forward(self, x, y_src, marginals=None):
        batch_size = x.shape[1]
        data_summary = self.summary_layer(x, y_src)
        if self.add_marginals:
            data_summary = torch.cat([data_summary, marginals], dim=-1)
        res = self.mlp(data_summary)

        if self.decoder_type in ["class_tokens", "class_average"]:
            # res is (batch, classes, n_features * n_bins + 1)
            if self.biattention:
                shape_functions = res
                biases = None
            else:
                shape_functions = res[:, :, :-1].reshape(batch_size, self.n_out, -1, self.n_bins)
                biases = res[:, :, -1]
            shape_functions = shape_functions.permute(0, 2, 3, 1)
        elif self.decoder_type == "average" and self.biattention:
            assert res.shape[2] == self.num_output_layer_weights
            shape_functions = res.reshape(batch_size, -1, self.n_bins, self.n_out)
            biases = None
        else:
            if self.biattention:
                shape_functions = res.reshape(batch_size, -1, self.n_bins, self.n_out)
                biases = None
            else:
                assert res.shape[1] == self.num_output_layer_weights
                shape_functions = res[:, :-self.n_out].reshape(batch_size, self.n_features, self.n_bins, self.n_out)
                biases = res[:, -self.n_out:]
        if not self.biattention:
            assert shape_functions.shape == (batch_size, self.n_features, self.n_bins, self.n_out)
            assert biases.shape == (batch_size, self.n_out)
        return shape_functions, biases


class FactorizedAdditiveModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_features=100, n_bins=64, n_out=10, hidden_size=1024, predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_hidden_layers=1, nhead=4,  weight_embedding_rank=None, rank=16, decoder_type="output_attention", shape_attention=False,
                 n_shape_functions=32, shape_attention_heads=1, shape_init="constant", biattention=False, decoder_activation='relu'):
        super().__init__()
        self.emsize = emsize
        self.n_features = n_features
        self.rank = rank
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank
        self.decoder_type = decoder_type
        self.shape_attention = shape_attention
        self.summary_layer = SummaryLayer(emsize=emsize, n_out=n_out, decoder_type=decoder_type, embed_dim=embed_dim, nhead=nhead)
        self.num_output_layer_weights = rank if biattention else rank * n_features
        self.n_shape_functions = n_shape_functions
        self.shape_attention_heads = shape_attention_heads
        self.shape_init = shape_init
        self.biattention = biattention
        self.decoder_activation = decoder_activation

        if decoder_type in ["class_tokens", "class_average"]:
            mlp_in_size = emsize
            # these serve as shared prototypes across features
            if shape_attention:
                if shape_init == "constant":
                    factor = 1
                elif shape_init == 'inverse':
                    factor = n_bins * n_features
                elif shape_init == 'sqrt':
                    factor = np.sqrt(n_bins * n_features)
                elif shape_init == 'inverse_bins':
                    factor = n_bins
                elif shape_init == 'inverse_sqrt_bins':
                    factor = np.sqrt(n_bins)
                else:
                    raise ValueError(f"Unknown shape_init: {shape_init}")

                self.shape_functions = nn.Parameter(torch.randn(n_shape_functions, n_bins) / factor)
                if shape_attention_heads == 1:
                    self.shape_function_keys = nn.Parameter(torch.randn(n_shape_functions, rank))
                else:
                    self.shape_function_keys = nn.ParameterList([nn.Parameter(torch.randn(n_shape_functions, rank)) for _ in range(shape_attention_heads)])
                    self.feature_heads = nn.ParameterList([nn.Parameter(torch.randn(rank, rank)) for _ in range(shape_attention_heads)])
                    self.head_weights = nn.Parameter(torch.randn(shape_attention_heads))
            else:
                self.output_weights = nn.Parameter(torch.randn(rank, n_bins))
        else:
            if shape_attention:
                raise ValueError("Shape attention is not supported for unless using class_tokens or class_average")
            mlp_in_size = self.summary_layer.out_size
            # these serve as shared prototypes across features and classes
            self.output_weights = nn.Parameter(torch.randn(rank, n_bins, n_out))

        self.mlp = make_decoder_mlp(mlp_in_size, hidden_size, self.num_output_layer_weights, n_layers=decoder_hidden_layers, activation=decoder_activation)
        self.output_biases = nn.Parameter(torch.randn(n_out))

    def forward(self, x, y_src, marginals=None):
        assert marginals is None
        summary = self.summary_layer(x, y_src)
        res = self.mlp(summary)
        if self.decoder_type in ["class_tokens", "class_average", "average"]:
            # for biattention, present features could be less than n_features
            res = res.reshape(x.shape[1], self.n_out, -1, self.rank)
            if self.shape_attention:
                if self.shape_attention_heads == 1:
                    out = nn.functional.scaled_dot_product_attention(res, self.shape_function_keys, self.shape_functions)
                    # reshape from batch, outputs, features, bins to batch, features, bins, outputs
                    out = out.permute(0, 2, 3, 1)
                else:
                    head_results = []
                    for key, head in zip(self.shape_function_keys, self.feature_heads):
                        head_results.append(nn.functional.scaled_dot_product_attention(res @ head, key, self.shape_functions))
                    head_results = torch.stack(head_results, dim=-1)
                    # b batch, k feature, r rank, o outputs, h heads, d bins
                    out = torch.einsum('bokdh, h -> bkdo', head_results, self.head_weights)
            else:
                out = torch.einsum('bokr, rd -> bkdo', res, self.output_weights)
        else:
            res = res.reshape(x.shape[1], self.n_features, self.rank)
            # b batch, k feature, r rank, o outputs, d bins
            out = torch.einsum('bkr, rdo -> bkdo', res, self.output_weights)
        return out, self.output_biases


class SummaryLayer(nn.Module):
    def __init__(self, emsize=512, embed_dim=2048, n_out=10, decoder_type='output_attention', nhead=4):
        super().__init__()
        self.emsize = emsize
        self.n_out = n_out
        self.decoder_type = decoder_type
        self.nhead = nhead

        if decoder_type == "output_attention":
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
            out_size = embed_dim
            self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)
        elif decoder_type == "special_token":
            out_size = emsize
            self.output_layer = nn.MultiheadAttention(embed_dim=emsize, num_heads=self.nhead)
        elif decoder_type == "special_token_simple":
            out_size = emsize
        elif decoder_type == "class_tokens":
            out_size = emsize * n_out
        elif decoder_type == "class_average":
            out_size = emsize * n_out
        elif decoder_type == "average":
            out_size = emsize
        else:
            raise ValueError(f"Unknown decoder_type {decoder_type}")

        self.out_size = out_size

    def forward(self, x, y_src):
        if x.shape[0] != 0:
            if self.decoder_type == "output_attention":
                if x.ndim == 3:
                    res = self.output_layer(self.query.repeat(1, x.shape[1], 1), x, x, need_weights=False)[0].squeeze(0)
                elif x.ndim == 4:
                    x_flat = x.reshape(x.shape[0], -1, x.shape[3])
                    res = self.output_layer(self.query.repeat(1, x_flat.shape[1], 1), x_flat, x_flat, need_weights=False)[0]
                    res = res.reshape(x.shape[1], x.shape[2], -1)
                else:
                    raise ValueError(f"Unknown x shape: {x.shape}")
            elif self.decoder_type == "special_token":
                res = self.output_layer(x[[0]], x[1:], x[1:], need_weights=False)[0]
            elif self.decoder_type == "special_token_simple":
                res = x[0]
            elif self.decoder_type == "class_tokens":
                res = x[:10].transpose(0, 1)
            elif self.decoder_type == "class_average":
                # per-class mean
                # clamping y_src to avoid -100 which should be ignored in loss
                # fingers crossed this will not mess things up
                y_src = y_src.long().clamp(0)
                # scatter add does not broadcast so we need to expand
                if y_src.ndim == 1:
                    # batch size 1
                    y_src = y_src.unsqueeze(1)
                if x.ndim == 3:
                    indices = y_src.unsqueeze(-1).expand(-1, -1, self.emsize)
                    sums = torch.zeros(self.n_out, x.shape[1], self.emsize, device=x.device, dtype = x.dtype)
                    sums.scatter_add_(0, indices, x)
                    # create counts
                    ones = torch.ones(1, device=x.device).expand(x.shape[0], x.shape[1])
                    counts = torch.zeros(self.n_out, x.shape[1], device=x.device)
                    indices = y_src
                elif x.ndim == 4:
                    indices = y_src.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], self.emsize)
                    sums = torch.zeros(self.n_out, x.shape[1], x.shape[2], self.emsize, device=x.device)
                    sums.scatter_add_(0, indices, x)
                    # create counts
                    ones = torch.ones(1, device=x.device).expand(x.shape[0], x.shape[1], x.shape[2])
                    counts = torch.zeros(self.n_out, x.shape[1], x.shape[2], device=x.device)
                    indices = y_src.unsqueeze(-1).expand(-1, -1, x.shape[2])
                else:
                    raise ValueError(f"Unknown x shape: {x.shape}")
                counts.scatter_add_(0, indices, ones)
                counts = counts.clamp(1e-10)  # don't divide by zero
                res = (sums / counts.unsqueeze(-1)).transpose(0, 1)
            elif self.decoder_type == "average":
                res = x.mean(0)
            else:
                raise ValueError(f"Unknown decoder_type: {self.decoder_type}")
        else:
            raise ValueError("Empty input")
        return res


def make_decoder_mlp(in_size, hidden_size, out_size, n_layers=1, activation='relu'):
    if activation == 'relu':
        activation = nn.ReLU
    elif activation == 'gelu':
        activation = nn.GELU
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return nn.Sequential(
        nn.Linear(in_size,  hidden_size),
        activation(),
        *sum([[nn.Linear(hidden_size, hidden_size), activation()] for _ in range(n_layers - 1)], []),
        nn.Linear(hidden_size, out_size))


class MLPModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_out=10, hidden_size=1024, decoder_type='output_attention', predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_hidden_layers=1, nhead=4, predicted_hidden_layers=1, weight_embedding_rank=None, low_rank_weights=False, decoder_activation='relu',
                 in_size=100):
        super().__init__()
        self.emsize = emsize
        self.embed_dim = embed_dim
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.decoder_type = decoder_type
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = in_size
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank if low_rank_weights else None

        self.predicted_hidden_layers = predicted_hidden_layers
        self.summary_layer = SummaryLayer(emsize=emsize, n_out=n_out, decoder_type=decoder_type, embed_dim=embed_dim, nhead=nhead)
        mlp_in_size = self.summary_layer.out_size
        if self.predicted_hidden_layers == 0:
            if decoder_type in ["class_tokens", "class_average"]:
                # class convolutional
                self.num_output_layer_weights = 1 + self.in_size
                mlp_in_size = emsize
            else:
                self.num_output_layer_weights =  n_out * (1 + self.in_size)
        elif self.weight_embedding_rank is None:
            self.num_output_layer_weights = (self.predicted_hidden_layer_size + 1) * n_out + (self.in_size + 1) * self.predicted_hidden_layer_size
            if self.predicted_hidden_layers > 1:
                self.num_output_layer_weights += (self.predicted_hidden_layers - 1) * (self.predicted_hidden_layer_size ** 2 + self.predicted_hidden_layer_size)
        else:
            self.num_output_layer_weights = (self.predicted_hidden_layer_size + 1) * n_out + self.in_size * \
                self.weight_embedding_rank + self.predicted_hidden_layer_size
            if self.predicted_hidden_layers > 1:
                self.num_output_layer_weights += (self.predicted_hidden_layers - 1) * (self.predicted_hidden_layer_size *
                                                                                       self.weight_embedding_rank + self.predicted_hidden_layer_size)
            self.shared_weights = nn.ParameterList([nn.Parameter(torch.randn(
                self.weight_embedding_rank, self.predicted_hidden_layer_size) / self.weight_embedding_rank) for _ in range(self.predicted_hidden_layers)])
        self.mlp = make_decoder_mlp(mlp_in_size, hidden_size, self.num_output_layer_weights, n_layers=decoder_hidden_layers,
                                    activation=decoder_activation)

    def forward(self, x, y_src):
        # x is samples x batch x emsize
        hidden_size = self.predicted_hidden_layer_size
        # summary layer goes from per-sample to per-dataset representations
        x_summary = self.summary_layer(x, y_src)
        if self.predicted_hidden_layers != 0 or self.decoder_type != "class_average":
            x_summary = x_summary.reshape(x.shape[1], self.summary_layer.out_size)
        res = self.mlp(x_summary)
        assert res.shape[-1] == self.num_output_layer_weights

        # let's confuse ourselves by storing them in the opposite order!
        def take_weights(res, shape):
            if len(shape) == 1:
                size = shape[0]
            elif len(shape) == 2:
                size = shape[0] * shape[1]
            else:
                raise ValueError("Only 1D and 2D shapes are supported")
            return res[:, :size].reshape(-1, *shape), res[:, size:]

        if self.predicted_hidden_layers == 0:
            if self.decoder_type == "class_average":
                w = res[:, :, :-1]
                w = w.transpose(1, 2)
                b = res[:, :, -1]
                return (b, w),
            else:
                w, next_res = take_weights(res, (self.in_size, self.n_out))
                b, next_res = take_weights(next_res, (self.n_out,))
                assert next_res.shape[1] == 0
                return (b, w), 
        if self.weight_embedding_rank is not None:
            second_shape = self.weight_embedding_rank
        else:
            second_shape = hidden_size

        w2, next_res = take_weights(res, (hidden_size, self.n_out))
        b2, next_res = take_weights(next_res, (self.n_out,))
        w1, next_res = take_weights(next_res, (self.in_size, second_shape))
        b1, next_res = take_weights(next_res, (hidden_size,))
        result = [(b1, w1)]
        for _ in range(self.predicted_hidden_layers - 1):
            w, next_res = take_weights(next_res, (hidden_size, second_shape))
            b, next_res = take_weights(next_res, (hidden_size,))
            result.append((b, w))
        assert next_res.shape[1] == 0
        result.append((b2, w2))

        return result

import torch
from torch import nn


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
                 decoder_two_hidden_layers=False, no_double_embedding=False, nhead=4, predicted_hidden_layers=1, weight_embedding_rank=None):
        super().__init__()
        self.emsize = emsize
        self.n_features = n_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.no_double_embedding = no_double_embedding
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100 if no_double_embedding else emsize
        out_size = emsize
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank

        self.predicted_hidden_layers = predicted_hidden_layers
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        out_size = embed_dim
        self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)
        self.num_output_layer_weights = n_out * (n_bins * n_features + 1)

        if decoder_two_hidden_layers:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))

    def forward(self, x):
        res = self.mlp(self.output_layer(self.query.repeat(1, x.shape[1], 1), x, x, need_weights=False)[0]).squeeze(0)
        assert res.shape[1] == self.num_output_layer_weights
        return res[:, :-self.n_out].reshape(-1, self.n_features, self.n_bins, self.n_out), res[:, -self.n_out:]


class FactorizedAdditiveModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_features=100, n_bins=64, n_out=10, hidden_size=1024, predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_two_hidden_layers=False, no_double_embedding=False, nhead=4, predicted_hidden_layers=1, weight_embedding_rank=None, rank=16):
        super().__init__()
        self.emsize = emsize
        self.n_features = n_features
        self.rank = rank
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.no_double_embedding = no_double_embedding
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100 if no_double_embedding else emsize
        out_size = emsize
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank

        self.predicted_hidden_layers = predicted_hidden_layers
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        out_size = embed_dim
        self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)
        self.num_output_layer_weights = rank * n_features

        if decoder_two_hidden_layers:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))
            
        # these serve as shared prototypes across features
        self.output_weights = nn.Parameter(torch.randn(rank, n_bins, n_out))
        self.output_biases = nn.Parameter(torch.randn(n_out))

    def forward(self, x):
        res = self.mlp(self.output_layer(self.query.repeat(1, x.shape[1], 1), x, x, need_weights=False)[0]).squeeze(0)
        res = res.reshape(x.shape[1], self.n_features, self.rank)
        # b batch, k feature, r rank, o outputs
        out = torch.einsum('bkr, rdo -> bkdo', res, self.output_weights)
        return out, self.output_biases
    


class MLPModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_out=10, hidden_size=1024, output_attention=False, special_token=False, predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_two_hidden_layers=False, no_double_embedding=False, nhead=4, predicted_hidden_layers=1, weight_embedding_rank=None):
        super().__init__()
        self.emsize = emsize
        self.embed_dim = embed_dim
        self.no_double_embedding = no_double_embedding
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.output_attention = output_attention
        self.special_token = special_token
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100 if no_double_embedding else emsize
        out_size = emsize
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank

        self.predicted_hidden_layers = predicted_hidden_layers

        if output_attention:
            if special_token:
                out_size = emsize
                self.output_layer = nn.MultiheadAttention(embed_dim=emsize, num_heads=self.nhead)

            else:
                self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
                out_size = embed_dim
                self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)

        if self.weight_embedding_rank is None:
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

        if decoder_two_hidden_layers:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(out_size,  hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_output_layer_weights))

    def forward(self, x):
        hidden_size = self.predicted_hidden_layer_size
        if x.shape[0] != 0:
            if self.output_attention:
                if not self.special_token:
                    res = self.mlp(self.output_layer(self.query.repeat(1, x.shape[1], 1), x, x, need_weights=False)[0]).squeeze(0)
                else:
                    res = self.mlp(self.output_layer(x[[-1]], x[:-1], x[:-1], need_weights=False)[0]).squeeze(0)
            else:
                res = self.mlp(x.mean(0))
        else:
            raise ValueError("Empty input")

        assert res.shape[1] == self.num_output_layer_weights
        # let's confuse ourselves by storing them in the opposite order!

        def take_weights(res, shape):
            if len(shape) == 1:
                size = shape[0]
            elif len(shape) == 2:
                size = shape[0] * shape[1]
            else:
                raise ValueError("Only 1D and 2D shapes are supported")
            return res[:, :size].reshape(-1, *shape), res[:, size:]
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

import torch
from torch import nn
import random


class ScaledDecoder(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super().__init__()
        self.linear = nn.Linear(ninp, nhid)
        self.linear1 = nn.Linear(nhid, nout)
        self.linear2 = nn.Linear(nhid, 10)

    def forward(self, x):
        #return torch.cat([self.linear1(x), self.linear2(x)], -1)
        x = self.linear(x)
        x = nn.GELU()(x)
        temps = self.linear2(x).softmax(-1) @ torch.tensor([1.,1.4,1.7,2.,5.,10.,20.,40.,80.,160.], device=x.device)
        if random.random() > .99:
            print(temps.shape,temps[:,:2])
        return self.linear1(x) / temps.unsqueeze(-1)

class FixedScaledDecoder(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super().__init__()
        self.mapper = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, nout))
        self.T = nn.Parameter(torch.ones(10000)/10000)

    def forward(self, x):
        return self.mapper(x)/self.T.sum()

class LinearModelDecoder(nn.Module):
    def __init__(self, emsize=512, nout=10, hidden_size=1024):
        super().__init__()
        self.emsize = emsize
        self.nout = nout
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(nn.Linear(emsize,  hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, (emsize + 1) * nout))

    def forward(self, x):
        return self.mlp(x.mean(0)).reshape(-1, self.emsize + 1, self.nout)


class MLPModelDecoder(nn.Module):
    def __init__(self, emsize=512, nout=10, hidden_size=1024, output_attention=False, special_token=False, predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_two_hidden_layers=False, no_double_embedding=False, nhead=4, predicted_hidden_layers=1):
        super().__init__()
        self.emsize = emsize
        self.embed_dim = embed_dim
        self.no_double_embedding = no_double_embedding
        self.nout = nout
        self.hidden_size = hidden_size
        self.output_attention = output_attention
        self.special_token = special_token
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        self.in_size = 100 if no_double_embedding else emsize
        out_size = emsize
        self.nhead = nhead
        self.predicted_hidden_layers = predicted_hidden_layers

        if output_attention:
            if special_token:
                out_size = emsize
                self.output_layer = nn.MultiheadAttention(embed_dim=emsize, num_heads=self.nhead)

            else:
                self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
                out_size = embed_dim
                self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)
    
        self.num_output_layer_weights = (self.predicted_hidden_layer_size + 1) * nout + (self.in_size + 1) * self.predicted_hidden_layer_size
        if self.predicted_hidden_layers > 1:
            self.num_output_layer_weights += (self.predicted_hidden_layers - 1) * (self.predicted_hidden_layer_size** 2 + self.predicted_hidden_layer_size)

        if decoder_two_hidden_layers:
            self.mlp = nn.Sequential(nn.Linear(out_size,  hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, self.num_output_layer_weights))
        else:
            self.mlp = nn.Sequential(nn.Linear(out_size,  hidden_size),
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
            res = torch.zeros((hidden_size + 1) * nout + (self.in_size + 1) * hidden_size, device=x.device)

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
        w2, next_res = take_weights(res, (hidden_size, self.nout))
        b2, next_res = take_weights(next_res, (self.nout,))
        w1, next_res = take_weights(next_res, (self.in_size, hidden_size))
        b1, next_res = take_weights(next_res, (hidden_size,))
        result = [(b1, w1)]
        for _ in range(self.predicted_hidden_layers - 1):
            w, next_res = take_weights(next_res, (hidden_size, hidden_size))
            b, next_res = take_weights(next_res, (hidden_size,))
            result.append((b, w))
        assert next_res.shape[1] == 0
        result.append((b2, w2))
        return result

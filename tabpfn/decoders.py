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
    def __init__(self, emsize=512, nout=10, hidden_size=1024, output_attention=False, special_token=False, predicted_hidden_layer_size=None):
        super().__init__()
        self.emsize = emsize
        self.nout = nout
        self.hidden_size = hidden_size
        self.output_attention = output_attention
        self.special_token = special_token
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize
        out_size = emsize
        if output_attention:
            if special_token:
                out_size = emsize
                self.output_layer = nn.MultiheadAttention(embed_dim=emsize, num_heads=4)

            else:
                embed_dim  = 2048

                self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
                out_size = embed_dim
                self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, kdim=emsize, vdim=emsize)
            
        self.mlp = nn.Sequential(nn.Linear(out_size,  hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, (self.predicted_hidden_layer_size + 1) * nout + (emsize + 1) * self.predicted_hidden_layer_size))


    def forward(self, x):
        emsize = self.emsize
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
            res = torch.zeros((hidden_size + 1) * nout + (emsize + 1) * hidden_size, device=x.device)
        w2_size = self.nout * hidden_size
        b2_size = self.nout
        w1_size = emsize * hidden_size
        b1_size = hidden_size
        assert res.shape[1] == w2_size + b2_size + w1_size + b1_size
        # let's confuse ourselves by storing them in the opposite order!
        w2 = res[:, :w2_size].reshape(-1, hidden_size, self.nout)
        b2 = res[:, w2_size: w2_size + b2_size].reshape(-1, self.nout)
        w1 = res[:, w2_size + b2_size: w2_size + b2_size + w1_size].reshape(-1, emsize, hidden_size)
        b1 = res[:, -b1_size:].reshape(-1, hidden_size)
        return b1, w1, b2, w2

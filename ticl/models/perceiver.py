from math import pi

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from ticl.models.decoders import MLPModelDecoder
from ticl.models.mothernet import MLPModelPredictor
from ticl.models.encoders import Linear

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class TabPerceiver(MLPModelPredictor):
    def __init__(
        self,
        *,
        nlayers,
        n_features,
        emsize=512,
        classification_task=True,
        input_axis=1,
        num_latents=512,
        cross_heads=1,
        nhead=8,
        cross_dim_head=64,
        latent_dim_head=64,
        n_out=10,
        attn_dropout=0.,
        dropout=0.,  # feed forward dropout
        self_per_cross_attn=1,
        decoder_hidden_size=512,
        predicted_hidden_layer_size=128,
        decoder_type="output_attention",
        decoder_embed_dim=512,
        decoder_hidden_layers=1,
        y_encoder_layer=None,
        predicted_hidden_layers=1,
        recompute_attn=None,  # ignored
        nhid_factor=None,  # ignored
        pre_norm=None,  # ignored
        efficient_eval_masking=None,  # ignored
        input_normalization=None,  # ignored
        low_rank_weights=None,
        y_encoder=None,  # ignored, y_encoder_layer is passed
        weight_embedding_rank=None,
        init_method=None,  # ignored
        tabpfn_zero_weights=None,  # ignored
        decoder_activation='relu',
        predicted_activation='relu',

    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          nlayers: Depth of net.
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).

          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.input_axis = input_axis
        # input_dim is the input to the transformer, which is after the first linear embedding, so it's emsize
        self.input_dim = emsize
        latent_dim = emsize
        self.decoder_type = decoder_type
        # FIXME cross heads one is too little!
        latent_heads = nhead
        self.n_out = n_out
        self.ff_dropout = dropout
        self.decoder_activation = decoder_activation
        assert decoder_type == "output_attention"
        self.predicted_activation = predicted_activation
        assert predicted_activation == "relu"
        self.latents = nn.Parameter(0.02 * torch.randn(num_latents, latent_dim))

        self.layers = nn.ModuleList([])
        for i in range(nlayers):
            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                latent_block = nn.Module()
                latent_block.add_module('latent_attn', PreNorm(latent_dim, Attention(
                    latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout)))
                latent_block.add_module('latent_ff', PreNorm(latent_dim, FeedForward(latent_dim, dropout=self.ff_dropout, mult=1)))
                self_attns.append(latent_block)

            cross_attn_layer = nn.Module()
            cross_attn_layer.add_module('cross_attn', PreNorm(latent_dim, Attention(latent_dim, emsize, heads=cross_heads,
                                        dim_head=cross_dim_head, dropout=attn_dropout), context_dim=emsize))
            cross_attn_layer.add_module('cross_ff', PreNorm(latent_dim, FeedForward(latent_dim, dropout=self.ff_dropout, mult=1)))
            cross_attn_layer.add_module('latents', self_attns)
            self.layers.append(cross_attn_layer)
        self.decoder = MLPModelDecoder(emsize=latent_dim, hidden_size=decoder_hidden_size, n_out=n_out, decoder_type=decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=latent_heads, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights, decoder_activation=decoder_activation,
                                       in_size=n_features)

    def inner_forward(self, data):
        # b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        # assert len(axis) == self.input_axis, 'input data must have the right number of axis'
        assert len(data.shape) == self.input_axis + 2, 'input data must have the right number of axis'
        b = data.shape[1]
        # concat to channels of data and flatten axis
        # data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        # attention is implemented with batch in first dimension
        data = rearrange(data, 'n b d -> b n d')

        # layers
        for layer in self.layers:
            x = layer.cross_attn(x, context=data) + x
            x = layer.cross_ff(x) + x

            for latent in layer.latents:
                x = latent.latent_attn(x) + x
                x = latent.latent_ff(x) + x

        x = rearrange(x, 'b n d -> n b d')
        return x

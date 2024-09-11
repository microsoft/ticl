import torch
from fast_transformers.events import EventDispatcher, QKVEvent
from fast_transformers.masking import FullMask, LengthMask

class linear_attention(torch.nn.Module):
    def __init__(
        self, 
        d_model, 
        n_heads, 
        eps = 1e-6, 
        default_mask = False,
        event_dispatcher = False,
):
        super(linear_attention, self).__init__()
        d_values = d_model // n_heads
        
        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)
        self.query_projection = torch.nn.Linear(d_model, d_values * n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_values * n_heads)
        self.value_projection = torch.nn.Linear(d_model, d_values * n_heads)
        
        self.n_heads = n_heads
        self.eps = eps
        self.default_mask = default_mask
        if event_dispatcher:
            self.event_dispatcher = EventDispatcher.get('')
        else:
            self.event_dispatcher = None
        
    def inner_attention(self, queries, keys, values, attn_mask = None, query_lengths = None, key_lengths = None):
        
        # Apply the feature map to the queries and keys
        Q = torch.nn.functional.elu(queries) + 1
        K = torch.nn.functional.elu(keys) + 1
        
        if self.default_mask:
            if not attn_mask.all_ones:
                raise RuntimeError(("LinearAttention does not support arbitrary "
                                    "attention masks"))
            K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


    def forward(self, queries, keys, values, attn_mask = None, query_lengths = None,
                key_lengths = None, is_causal = False, **kwargs):
        # Extract the dimensions into local variables
        if is_causal: raise NotImplementedError("Causal attention is not implemented for linear attention")
        
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)
        
        if self.default_mask:
            attn_mask = attn_mask or FullMask(N, device=queries.device)
            length_mask = LengthMask(queries.new_full((N,), L, dtype=torch.int64))
        else:
            attn_mask = None
            length_mask = None
        
        if self.event_dispatcher: 
            self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask = attn_mask,
            query_lengths = length_mask,
            key_lengths = length_mask,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)
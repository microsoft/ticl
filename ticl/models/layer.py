from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules.transformer import (Dropout, LayerNorm, Linear, Module, Optional, Tensor,
                                          _get_activation_fn)
from torch.utils.checkpoint import checkpoint

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module

class BiAttentionEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.cross_sample_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        post_feature_attention = self.cross_feature_attention(src.reshape(-1, *src.shape[2:]).transpose(0, 1), src_mask)
        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask)
        return res.reshape(src.shape)


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation="relu",
        layer_norm_eps=1e-5, 
        batch_first=True, 
        pre_norm=False,
        device=None, 
        dtype=None, 
        recompute_attn=False,
        attn_name = 'default',
        feature_map='identity',
        norm_output = False,
    ) -> None:
        # batch_first is set to True for using flash attention II
        # check the details of when flash attention can be triggered here: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if attn_name == 'fla': attn_name = 'flash_linear_attention'

        if (torch.__version__ >= '2.2.0') or (not attn_name in ['default', 'flash_attention']):
            if attn_name == 'default': attn_name = 'flash_attention'
            from ticl.models.flash_transformer import MultiheadAttention
            self.self_attn = MultiheadAttention(
                d_model, 
                nhead, 
                dropout=dropout, 
                batch_first=batch_first,
                attn_name = attn_name,
                feature_map = feature_map,
                norm_output = norm_output,
                **factory_kwargs,
            )
        else: 
            # cannot use 'flash_attention'
            from torch.nn import MultiheadAttention
            self.self_attn = MultiheadAttention(
                d_model, 
                nhead, 
                dropout=dropout, 
                batch_first=batch_first,
                **factory_kwargs,
            )
        self.attn_name = attn_name
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn

        self.activation = _get_activation_fn(activation)

    def forward(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None, 
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

        if isinstance(src_mask, tuple):
            return NotImplementedError
        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None

            single_eval_position = src_mask
            if is_causal:
                if self.attn_name in ['flash_attention', 'default']:
                    train_mask = nn.Transformer.generate_square_subsequent_mask(single_eval_position).to(dtype=torch.bool,device=src.device)
                elif self.attn_name == 'flash_linear_attention':
                    # the causal mask is implemented internally
                    train_mask = None
                test_mask = None
                train_is_causal = True
                test_is_causal = False
            else:
                train_mask = None
                test_mask = None
                train_is_causal = False
                test_is_causal = False

            # split the training and testing samples
            src_train = src_[:single_eval_position]
            src_test = src_[single_eval_position:]
            # since we set batch_first = True, the shape of src_ is (batch, seq, feature)
            src_train = src_train.permute(1, 0, 2)
            src_test = src_test.permute(1, 0, 2)

            # the training samples are only attend to themselves
            src_left = self.self_attn(
                src_train, 
                src_train, 
                src_train, 
                attn_mask=train_mask,
                is_causal=train_is_causal,
                need_weights=False,
            )[0]

            # the testing samples attend to training samples
            src_right = self.self_attn(
                src_test, 
                src_train, 
                src_train,
                attn_mask=test_mask,
                is_causal=test_is_causal,
                need_weights=False,
            )[0]

            # permute them back to (seq, batch, feature)
            src_left = src_left.permute(1, 0, 2)
            src_right = src_right.permute(1, 0, 2)
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                # this might have some problems, double check
                # https://github.com/pytorch/pytorch/issues/99282
                src2 = checkpoint(
                    self.self_attn, 
                    src_, 
                    src_, 
                    src_, 
                    src_key_padding_mask, 
                    # need_weights if specified returns attn_output_weights 
                    # in addition to attn_outputs
                    True, 
                    src_mask,
                    True,
                    is_causal,
                )[0]
            else:
                src2 = self.self_attn(
                    query = src_, 
                    key = src_, 
                    value = src_, 
                    attn_mask = src_mask,
                    key_padding_mask = src_key_padding_mask,
                    is_causal = is_causal,
                    need_weights=False,
                )[0]
        
        # residual connection
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src

def get_ssm_layers(
    d_model: int,
    n_layer: int,
    d_intermediate: int,
    model = 'mamba1',
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    initializer_cfg=None,
    fused_add_norm=False,
    residual_in_fp32=False,
    device=None,
    dtype=None,
    nheads = 2,
    dropout = 0.0,
    activation = 'gelu',
    pre_norm = False,
    recompute_attn = False,
    all_layers_same_init = False,
    norm_output = False,
    feature_map = 'identity',
):
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if 'mamba' in model:
        from ticl.models.mamba import MambaLayer
        if ssm_cfg is None:
            ssm_cfg = {
                'layer': model[0].upper()+model[1:].lower(),
            }
        else:
            ssm_cfg = {
                'layer': model[0].upper()+model[1:].lower(), 
                **ssm_cfg,
            }
        return MambaLayer(
            d_model,
            n_layer,
            d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype,
        )
    elif model == 'linear_attention':
        from ticl.models.linear_attention import TransformerEncoderBuilder

        if d_model % nheads != 0:
            raise ValueError(f"nheads {nheads} must divide d_model {d_model}!")

        # Create the builder for our transformers
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layer,
            n_heads=nheads,
            query_dimensions=d_model // nheads,
            value_dimensions=d_model // nheads,
            feed_forward_dimensions=d_intermediate,
        )

        # Build a transformer with linear attention
        builder.attention_type = "linear"
        linear_model = builder.get()

        return linear_model
    
    elif model in ['fla', 'retnet']:
        from torch.nn import TransformerEncoder
        from ticl.models.tabpfn import TransformerEncoderDiffInit
        # import os 
        # os.environ['CUDA_LAUNCH_BLOCKING']="1"
        # os.environ['TORCH_USE_CUDA_DSA'] = "1"          

        if model == 'fla': 
            attn_name = 'fla'
        elif model == 'retnet':
            attn_name = 'retention'
        
        def encoder_layer_creator(): return TransformerEncoderLayer(
            d_model, 
            nheads,
            d_intermediate,
            dropout, 
            activation = activation, 
            pre_norm = pre_norm, 
            recompute_attn = recompute_attn,  
            attn_name = model,
            norm_output = norm_output,
            feature_map = feature_map,
        )
        transformer_encoder = TransformerEncoder(
            encoder_layer_creator(),
            n_layer,
        ) if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, n_layer)

        return transformer_encoder
    elif model == 'gla':
        pass
    elif model == 'retnet':
        pass
    else:
        raise ValueError(f"Unknown model {model}")

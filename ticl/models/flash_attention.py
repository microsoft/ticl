# -*- coding: utf-8 -*-

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import RMSNorm
from fla.modules.feature_map import (DPFPFeatureMap, HadamardFeatureMap,
                                     HedgehogFeatureMap, T2RFeatureMap)
from fla.ops.linear_attn import (chunk_linear_attn, fused_chunk_linear_attn,
                                 fused_recurrent_linear_attn)


class FlashLinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        # standard linear attention normalization
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        proj_bias: bool = False,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                # self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                # self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_q = HadamardFeatureMap(head_dim=self.key_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.key_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            # self.feature_map_q = nn.Identity()
            # self.feature_map_k = nn.Identity()
            pass
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=proj_bias)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=proj_bias)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=proj_bias)

        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            pass
            # self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    # the parameters are set to be identical to normal MultiheadAttention for compatibility
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    def forward(
        self, 
        query, 
        key, 
        value,
        key_padding_mask = None,
        need_weights = True, 
        attn_mask = None,
        average_attn_weights = True,
        is_causal = False, 
    ):
        # x: [B, L, hidden_size]
        mode = self.mode
        # q,k: [B, L, key_dim], where key_dim = hidden_size * expand_k
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if hasattr(self, 'feature_map_q'):
            # old: head_qk_dim, where head_qk_dim = key_dim // num_heads
            # new: key_dim
            q = self.feature_map_q(q)
            k = self.feature_map_k(k)

        # q: [B, num_heads, L, head_qk_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v = (repeat(x, 'b n (h d) -> b (h g) n d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, 'b n (h d) -> b h n d', h=self.num_kv_heads) for x in (k, v))
        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)

        # kernel: a function that is meant to be executed in parallel on an attached GPU is called a kernel
        # Fusion in this context refers to combining several different operations (like arithmetic operations or function applications) into one kernel function. 
        # Instead of launching separate kernels for each operation (each with its setup and teardown overhead), a single kernel does all the work in one go.
        if mode == 'chunk':
            attn_weights, final_state = chunk_linear_attn(q, k, v, normalize=self.do_feature_map_norm)
        elif mode == 'fused_chunk':
            attn_weights, final_state = fused_chunk_linear_attn(q, k, v, normalize=self.do_feature_map_norm)
        elif mode == 'fused_recurrent':
            attn_weights, final_state = fused_recurrent_linear_attn(q, k, v, normalize=self.do_feature_map_norm)
        else:
            raise NotImplementedError
        
        if hasattr(self, 'norm'):
            o = self.norm(attn_weights)
        else:
            o = attn_weights

        # combine different heads
        o = rearrange(o, 'b h n d -> b n (h d)')
        o = self.o_proj(o)
        return o, attn_weights

from typing import Optional
from torch import Tensor
from torch.nn import Module
import torch 
from torch.nn.modules.transformer import _get_clones, _get_seq_len, _detect_is_causal_mask
import warnings

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output


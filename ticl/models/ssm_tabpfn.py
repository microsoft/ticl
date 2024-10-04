import torch, wandb
import torch.nn as nn 

import numpy as np

from ticl.models.encoders import Linear
from ticl.models.layer import get_ssm_layers

from ticl.utils import SeqBN

import pdb

class SSMTabPFN(nn.Module):
    def __init__(
        self, 
        *, 
        model, 
        n_out, 
        emsize, 
        nhead, 
        nhid_factor, 
        nlayers, 
        n_features, 
        dropout=0.0,  
        y_encoder_layer=None,
        decoder=None, 
        input_normalization=False, 
        init_method=None, 
        pre_norm=False,
        activation='gelu', 
        recompute_attn=False, 
        classification_task=True,
        all_layers_same_init=False, 
        efficient_eval_masking=True, 
        y_encoder=None, 
        tabpfn_zero_weights=False,
        local_nhead=4, 
        causal_mask = False, 
        norm_output = False,
        feature_map = 'identity',
        ssm_cfg={},
    ):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor
        self.model = model

        # def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
        #                                                             pre_norm=pre_norm, recompute_attn=recompute_attn)
        # self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
        #     if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        
        self.causal_mask = causal_mask

        self.ssm = get_ssm_layers(
            d_model = emsize,
            n_layer = nlayers,
            d_intermediate = nhid,
            model = model,
            nheads = nhead,
            ssm_cfg = ssm_cfg,
            norm_output = norm_output,
            feature_map = feature_map,
        )
        backbone_size = sum(p.numel() for p in self.ssm.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        
        if self.model == 'mamba1': x_src = x_src.contiguous()
        x_src = self.encoder(x_src) # transform n_features to emsize
        # transform y as one-hot encoding into emsize
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        # What does src_mask do here?
        if src_mask is None:
            full_len = len(x_src)
            if self.efficient_eval_masking:
                src_mask = single_eval_pos # tell us when to cut the sequence
            else:
                src_mask = self.generate_D_q_matrix(full_len, len(x_src) - single_eval_pos).to(x_src.device)

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0) # concatenate the training sequence and the test point
        # we have many testing point...

        if self.input_ln is not None:
            src = self.input_ln(src)
            
        # output = self.ssm(src, src_mask)
        if self.model in ['linear_attention', 'mamba1', 'mamba2']:
            output = self.ssm(src, src_mask)
        elif self.model in ['fla']:
            output = self.ssm(src, src_mask, is_causal = self.causal_mask)
        else:
            raise NotImplementedError(f"Model {self.model} is not implemented yet.")
        
        output = self.decoder(output) 
        return output[single_eval_pos:]
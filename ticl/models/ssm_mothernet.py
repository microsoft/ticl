import torch, wandb
import torch.nn as nn

from ticl.utils import SeqBN

from ticl.models.decoders import MLPModelDecoder
from ticl.models.mothernet import MLPModelPredictor
from ticl.models.encoders import Linear
from ticl.models.layer import get_ssm_layers

class SSMMotherNet(MLPModelPredictor):
    def __init__(self, *, model, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_type="output_attention", predicted_hidden_layer_size=None,
                 decoder_embed_dim=2048, classification_task=True,
                 decoder_hidden_layers=1, decoder_hidden_size=None, predicted_hidden_layers=1, weight_embedding_rank=None, y_encoder=None,
                 low_rank_weights=False, tabpfn_zero_weights=True, decoder_activation="relu", predicted_activation="relu",
                 local_nhead=4, ssm_cfg={}):
        super().__init__()
        self.classification_task = classification_task
        nhid = emsize * nhid_factor 
        
        self.ssm = get_ssm_layers(
            d_model = emsize,
            n_layer = nlayers,
            d_intermediate = nhid,
            model = model,
            nheads = nhead, 
            ssm_cfg = ssm_cfg,
        )
        backbone_size = sum(p.numel() for p in self.ssm.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)
        
        self.decoder_activation = decoder_activation
        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.y_encoder = y_encoder_layer 
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method # for the weight initialization 
        self.efficient_eval_masking = efficient_eval_masking # ?
        self.n_out = n_out
        self.nhid = nhid
        self.decoder_type = decoder_type # how did you create MLP from the transformer
        # average for each class 
        decoder_hidden_size = decoder_hidden_size or nhid # ? mlp? 
        self.tabpfn_zero_weights = tabpfn_zero_weights # ? 
        self.predicted_activation = predicted_activation # ? 

        self.decoder = MLPModelDecoder(emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out, decoder_type=self.decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights, decoder_activation=decoder_activation,
                                       in_size=n_features)
        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))

    def inner_forward(self, train_x):
        return self.ssm(train_x)
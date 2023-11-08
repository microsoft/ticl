import numpy as np


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder

from tabpfn.layer import TransformerEncoderLayer
from tabpfn.utils import SeqBN, bool_mask_to_att_mask
from tabpfn.utils import normalize_by_used_features_f, normalize_data

from tabpfn.transformer import TransformerEncoderDiffInit
from tabpfn.transformer_make_model import TransformerModelMaker
from tabpfn.encoders import Linear
from tabpfn.decoders import AdditiveModelDecoder
from tabpfn.scripts.model_builder import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


class MotherNetAdditive(nn.Module):
    def __init__(self, n_features, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, y_encoder=None,
                 pos_encoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_embed_dim=2048,
                 decoder_two_hidden_layers=False, decoder_hidden_size=None, n_bins=64):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = Linear(num_features=n_features*n_bins, emsize=ninp, replace_nan_by_zero=True)
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = AdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=ninp, hidden_size=nhid, n_out=n_out)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.init_method = init_method
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking
        self.n_bins = n_bins
        self.n_out = n_out
        self.nhid = nhid

        self.decoder = AdditiveModelDecoder(emsize=ninp, hidden_size=decoder_hidden_size, n_out=n_out,
                                            embed_dim=decoder_embed_dim,
                                            decoder_two_hidden_layers=decoder_two_hidden_layers, nhead=nhead)

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        # ?!?!? FIXME THIS SEEMS WRONG
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:,train_size:].zero_()
        mask[:,train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        #mask[:,num_global_att_tokens:].zero_()
        #mask[:,num_global_att_tokens:] |= torch.eye(trainset_size) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        mask = torch.zeros(num_global_att_tokens, num_global_att_tokens+seq_len-num_query_tokens) == 0
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        _, x_src_org, y_src = src
        X_onehot, _ = bin_data(x_src_org, n_bins=self.n_bins)
        X_onehot_flat = X_onehot.reshape((*X_onehot.shape[:-2], -1)).float()
        x_src = self.encoder(X_onehot_flat)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]

        output = self.transformer_encoder(train_x)
        weights, biases = self.decoder(output)

        h = (X_onehot[single_eval_pos:].unsqueeze(-1) * weights.unsqueeze(0)).sum([2, 3])
        h = h + biases

        if h.isnan().all():
            print("NAN")
            import pdb; pdb.set_trace()
        return h


def bin_data(data, n_bins):
    # FIXME treat NaN as separate bin
    data_nona = torch.nan_to_num(data, nan=0)
    quantiles = torch.arange(n_bins, device=data.device) / (n_bins - 1)
    bin_edges = torch.quantile(data_nona, quantiles, dim=0)
    # FIXME extra data copy
    bin_edges = bin_edges.transpose(0, -1).contiguous()
    data_nona = data_nona.transpose(0, -1).contiguous()
    X_binned = torch.searchsorted(bin_edges, data_nona)
    X_onehot = torch.nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    return X_onehot, bin_edges

def extract_additive_model(model, X_train, y_train, device="cpu", inference_device="cpu"):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train).to(device)
    xs = torch.Tensor(X_train).to(device)

    eval_xs_ = normalize_data(xs, eval_position)

    eval_xs = normalize_by_used_features_f(eval_xs_, X_train.shape[-1], max_features,
                                                   normalize_with_sqrt=False)
    if X_train.shape[1] > 100:
        raise ValueError("Cannot run inference on data with more than 100 features")
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)

    X_onehot = bin_data(x_all_torch, n_bins=self.n_bins)
    X_onehot_flat = X_onehot.reshape((*X_onehot.shape[:-2], -1)).float()
    # why need :len?
    x_src = model.encoder(X_onehot_flat.unsqueeze(1)[:len(X_train)])
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    output = model.transformer_encoder(train_x)

    weights, biases = model.decoder(output)
    import pdb; pdb.set_trace()
    w_data_space_prenorm = weights.squeeze()[:n_features, :]
    b_data_space = biases.squeeze()
    w_data_space = w_data_space_prenorm / (n_features / max_features)

    # remove extra classes on output layer
    w, b = w_data_space.squeeze()[:, :n_classes], b_data_space.squeeze()[:n_classes]

    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()

    return detach(b), detach(w)


def predict_with_mlp_model(X_train, X_test, layers, inference_device="cpu"):
    if inference_device == "cpu":
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0, ddof=1) + .000001
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[np.isnan(std)] = 1
        X_test_scaled = (X_test - mean) / std
        out = np.clip(X_test_scaled, a_min=-100, a_max=100)
        for i, (b, w) in enumerate(layers):
            out = np.dot(out, w) + b
            if i != len(layers) - 1:
                out = np.maximum(out, 0)
        if np.isnan(out).any():
            print("NAN")
            import pdb; pdb.set_trace()
        from scipy.special import softmax
        return softmax(out / .8, axis=1)
    elif "cuda" in inference_device:
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class ForwardMLPModel(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", label_offset=0, inference_device="cpu"):
        self.path = path or "models_diff/prior_diff_real_checkpoint_predict_mlp_nlayer12_multiclass_04_13_2023_16_41_16_n_0_epoch_37.cpkt"
        self.device = device
        self.label_offset = label_offset
        self.inference_device = inference_device

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        model, config = load_model(self.path, device=self.device)
        if "model_maker" not in config:
            raise ValueError("Cannot load tabpfn weights into ForwardMLPModel")
        if config['model_maker'] != "mlp":
            raise ValueError(f"Incompatible model_maker: {config['model_maker']}")
        model.to(self.device)
        n_classes = len(le.classes_)
        indices = np.mod(np.arange(n_classes) + self.label_offset, n_classes)
        layers = extract_mlp_model(model, X, np.mod(y + self.label_offset, n_classes), device=self.device, inference_device=self.inference_device)
        if self.label_offset == 0:
            self.parameters_ = layers
        else:
            *lower_layers, b_last, w_last = layers
            self.parameters_  = (*lower_layers, (b_last[indices], w_last[:, indices]))
        self.classes_ = le.classes_
        return self

    def predict_proba(self, X):
        return predict_with_mlp_model(self.X_train_, X, self.parameters_, inference_device=self.inference_device)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
import numpy as np
import itertools
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder

from tabpfn.layer import TransformerEncoderLayer
from tabpfn.utils import SeqBN, bool_mask_to_att_mask
from tabpfn.utils import normalize_by_used_features_f, normalize_data

from tabpfn.transformer import TransformerEncoderDiffInit
from tabpfn.decoders import LinearModelDecoder, MLPModelDecoder
from tabpfn.scripts.model_builder import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

from einops import rearrange, repeat


class TransformerModelMaker(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = LinearModelDecoder(emsize=ninp, hidden_size=nhid, nout=n_out)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.n_out = n_out
        self.nhid = nhid

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
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
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

        if len(src) == 2: # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        style_src = self.style_encoder(style_src).unsqueeze(0) if self.style_encoder else \
            torch.tensor([], device=x_src.device)

        if single_eval_pos == 0:
            linear_model_coefs = torch.zeros((x_src.shape[1], x_src.shape[2] + 1, self.n_out), device=x_src.device)
        else:
            train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
            # src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)
            output = self.transformer_encoder(train_x)

            linear_model_coefs = self.decoder(output)
        matmul = (x_src[single_eval_pos:].unsqueeze(-1) * linear_model_coefs[:, :-1].unsqueeze(0)).sum(2)
        result = matmul + linear_model_coefs[:, -1]
        if result.isnan().all():
            import pdb; pdb.set_trace()
        return result
    
    
class TransformerModelMakeMLP(TransformerModelMaker):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, output_attention=False, special_token=False, predicted_hidden_layer_size=None, decoder_embed_dim=2048,
                 decoder_two_hidden_layers=False, decoder_hidden_size=None, no_double_embedding=False, predicted_hidden_layers=1, weight_embedding_rank=None):
        super().__init__(encoder, n_out, ninp, nhead, nhid, nlayers, dropout=dropout, style_encoder=style_encoder, y_encoder=y_encoder,
                 pos_encoder=pos_encoder, input_normalization=input_normalization, init_method=init_method, pre_norm=pre_norm,
                 activation=activation, recompute_attn=recompute_attn, num_global_att_tokens=num_global_att_tokens, full_attention=full_attention,
                 all_layers_same_init=all_layers_same_init, efficient_eval_masking=efficient_eval_masking)
        self.no_double_embedding = no_double_embedding
        self.output_attention = output_attention
        self.special_token = special_token
        self.decoder = MLPModelDecoder(emsize=ninp, hidden_size=decoder_hidden_size, nout=n_out, output_attention=self.output_attention,
                                       special_token=special_token, predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_two_hidden_layers=decoder_two_hidden_layers, no_double_embedding=no_double_embedding, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank)
        if special_token:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, ninp))

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2: # (x,y) and no style
            src = (None,) + src

        _, x_src_org, y_src = src
        x_src = self.encoder(x_src_org)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        if self.special_token:
            train_x = torch.cat([self.token_embedding.repeat(1, train_x.shape[1], 1), train_x], 0)
        
        output = self.transformer_encoder(train_x)
        (b1, w1), *layers = self.decoder(output)
        
        if self.no_double_embedding:
            x_src_org_nona = torch.nan_to_num(x_src_org[single_eval_pos:], nan=0)
            h = (x_src_org_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)
        else:
            h = (x_src[single_eval_pos:].unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

        if self.decoder.weight_embedding_rank is not None:
            h = torch.matmul(h, self.decoder.shared_weights[0]) 
        h = h + b1
        
        for i, (b, w) in enumerate(layers):
            h = torch.relu(h)
            h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
            if self.decoder.weight_embedding_rank is not None and i != len(layers) - 1:
                # last layer has no shared weights
                h = torch.matmul(h, self.decoder.shared_weights[i + 1])
            h = h + b
        
        if h.isnan().all():
            print("NAN")
            import pdb; pdb.set_trace()
        return h


def extract_linear_model(model, X_train, y_train, device="cpu"):
    max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train).to(device)
    xs = torch.Tensor(X_train).to(device)

    eval_xs_ = normalize_data(xs, eval_position)

    eval_xs = normalize_by_used_features_f(eval_xs_, X_train.shape[-1], max_features,
                                                   normalize_with_sqrt=False)
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)
    
    x_src = model.encoder(x_all_torch.unsqueeze(1)[:len(X_train)])
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    # src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)
    output = model.transformer_encoder(train_x)
    linear_model_coefs = model.decoder(output)
    encoder_weight = model.encoder.get_parameter("weight")
    encoder_bias = model.encoder.get_parameter("bias")

    total_weights = torch.matmul(encoder_weight[:, :n_features].T, linear_model_coefs[0, :-1, :n_classes])
    total_biases = torch.matmul(encoder_bias, linear_model_coefs[0, :-1, :n_classes]) + linear_model_coefs[0, -1, :n_classes]
    return total_weights.detach().cpu().numpy() / (n_features / max_features), total_biases.detach().cpu().numpy()



def extract_mlp_model(model, X_train, y_train, device="cpu", inference_device="cpu"):
    if inference_device == "cuda" and device == "cpu":
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
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)
    
    x_src = model.encoder(x_all_torch.unsqueeze(1)[:len(X_train)])
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    if hasattr(model, "transformer_encoder"):
        # tabpfn mlp model maker
        output = model.transformer_encoder(train_x)
    else:
        # perceiver
        data = rearrange(train_x, 'n b d -> b n d')
        x = repeat(model.latents, 'n d -> b n d', b = data.shape[0])

        # layers
        for cross_attn, cross_ff, self_attns in model.layers:
            x = cross_attn(x, context = data) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        output = rearrange(x, 'b n d -> n b d')
    (b1, w1), *layers = model.decoder(output)

    if model.no_double_embedding:
        w1_data_space_prenorm = w1.squeeze()[:n_features, :]
        b1_data_space = b1.squeeze()
    else:
        encoder_weight = model.encoder.get_parameter("weight")
        encoder_bias = model.encoder.get_parameter("bias")

        w1_data_space_prenorm  = torch.matmul(encoder_weight[:, :n_features].T, w1)
        b1_data_space = torch.matmul(encoder_bias, w1) + b1
    
    w1_data_space = w1_data_space_prenorm / (n_features / max_features)

    if model.decoder.weight_embedding_rank is not None:
        w1_data_space = torch.matmul(w1_data_space, model.decoder.shared_weights[0])

    layers_result = [(b1_data_space, w1_data_space)]

    for i, (b, w) in enumerate(layers[:-1]):
        if model.decoder.weight_embedding_rank is not None:
            w = torch.matmul(w, model.decoder.shared_weights[i + 1])
        layers_result.append((b, w))
    
    # remove extra classes on output layer
    layers_result.append((layers[-1][0].squeeze()[:n_classes], layers[-1][1].squeeze()[:, :n_classes]))

    if inference_device == "cpu":
        def detach(x):
            return x.squeeze().detach().cpu().numpy()
    else:
        def detach(x):
            return x.squeeze().detach()
    
    return [(detach(b), detach(w)) for (b, w) in layers_result]
    
def predict_with_linear_model(X_train, X_test, weights, biases):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1) + .000001
    X_test_scaled = (X_test - mean) / std
    X_test_scaled = np.clip(X_test_scaled, a_min=-100, a_max=100)
    res2 = np.dot(X_test_scaled , weights) + biases
    from scipy.special import softmax
    return softmax(res2 / .8, axis=1)


def predict_with_linear_model_complicated(model, X_train, y_train, X_test):
    max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train)
    xs = torch.Tensor(np.vstack([X_train, X_test]))

    eval_xs_ = normalize_data(xs, eval_position)

    eval_xs = normalize_by_used_features_f(eval_xs_, X_train.shape[-1], max_features,
                                                   normalize_with_sqrt=False)
    x_all_torch = torch.Tensor(np.hstack([eval_xs, np.zeros((eval_xs.shape[0], 100 - eval_xs.shape[1]))]))
    
    x_src = model.encoder(x_all_torch.unsqueeze(1)[:len(X_train)])
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    # src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)
    output = model.transformer_encoder(train_x)
    linear_model_coefs = model.decoder(output)
    encoder_weight = model.encoder.get_parameter("weight")
    encoder_bias = model.encoder.get_parameter("bias")

    total_weights = torch.matmul(encoder_weight[:, :n_features].T, linear_model_coefs[0, :-1, :n_classes])
    total_biases = torch.matmul(encoder_bias, linear_model_coefs[0, :-1, :n_classes]) + linear_model_coefs[0, -1, :n_classes]
                      
    pred_simple = torch.matmul(model.encoder(x_all_torch),  linear_model_coefs[0, :-1, :n_classes]) + linear_model_coefs[0, -1, :n_classes]
    probs =  torch.nn.functional.softmax(pred_simple/ 0.8, dim=1)
    return total_weights.detach().numpy() / (n_features / max_features), total_biases.detach().numpy(), probs[eval_position:]


from sklearn.base import BaseEstimator, ClassifierMixin


class ForwardLinearModel(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu"):
        self.path = path or "models_diff/prior_diff_real_checkpoint_predict_linear_coefficients_nlayer_6_multiclass_04_11_2023_01_26_19_n_0_epoch_94.cpkt"
        self.device = device
        
    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        model, _ = load_model(self.path, device=self.device)

        weights, biases = extract_linear_model(model, X, y, device=self.device)
        self.weights_ = weights
        self.biases_ = biases
        self.classes_ = le.classes_
        return self
        
    def predict_proba(self, X):
        return predict_with_linear_model(self.X_train_, X, self.weights_, self.biases_)
    
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


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
    elif inference_device == "cuda":
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        for (b, w) in layers:
            out = torch.relu(torch.matmul(out, w) + b)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()


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
        model, _ = load_model(self.path, device=self.device)
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


class PermutationsMeta(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, y):
        estimators = []
        for i in range(len(np.unique(y))):
            estimator = clone(self.base_estimator).set_params(label_offset=i)
            estimators.append((str(i), estimator))
        self.vc_ = VotingClassifier(estimators, voting='soft')
        self.vc_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.vc_.predict_proba(X)
    
    def predict(self, X):
        return self.vc_.predict(X)

    @property
    def classes_(self):
        return self.vc_.classes_

class ShiftClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, feature_shift=0, label_shift=0, transformer=None):
        self.base_estimator = base_estimator
        self.feature_shift = feature_shift
        self.label_shift = label_shift
        self.transformer = transformer

    def _feature_shift(self, X):
        return np.concatenate([X[:, self.feature_shift:], X[:, :self.feature_shift]], axis=1)

    def fit(self, X, y):
        if self.transformer is not None:
            X = self.transformer.fit_transform(X)
        X = self._feature_shift(X)
        unique_y = np.unique(y)
        self.n_classes_ = len(np.unique(y))
        self.class_indices_ = np.mod(np.arange(self.n_classes_) + self.label_shift, self.n_classes_)

        if not (unique_y == np.arange(self.n_classes_)).all():
            raise ValueError('y has to be in range(0, n_classes) but is %s' % unique_y)
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, np.mod(y + self.label_shift, self.n_classes_))
        return self
    
    def predict_proba(self, X):
        X = self._feature_shift(X)
        if self.transformer is not None:
            X = self.transformer.transform(X)
        return self.base_estimator_.predict_proba(X)[:, self.class_indices_]
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

class EnsembleMeta(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, n_estimators=32, random_state=0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        shifts = list(itertools.product(np.arange(self.n_classes_), np.arange(self.n_features_), [True, False]))
        rng = random.Random(self.random_state)
        shifts = rng.sample(shifts, min(len(shifts), self.n_estimators))
        estimators = []
        for label_shift, feature_shift, use_power_transformer in shifts:
            estimator = ShiftClassifier(self.base_estimator, feature_shift=feature_shift, label_shift=label_shift)
            if use_power_transformer:
                estimator = Pipeline([('power_transformer', PowerTransformer()), ('shift_classifier', estimator)])
            estimators.append((str((label_shift, feature_shift, use_power_transformer)), estimator))
        self.vc_ = VotingClassifier(estimators, voting='soft')
        self.vc_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.vc_.predict_proba(X)
    
    def predict(self, X):
        return self.vc_.predict(X)

    @property
    def classes_(self):
        return self.vc_.classes_

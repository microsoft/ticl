import itertools
import random

import numpy as np
import torch
from einops import rearrange, repeat
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

from mothernet.model_builder import load_model
from mothernet.utils import normalize_by_used_features_f, normalize_data


def extract_linear_model(model, X_train, y_train, device="cpu"):
    max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train).to(device)
    xs = torch.Tensor(X_train).to(device)

    eval_xs_ = normalize_data(xs, eval_position)

    eval_xs = normalize_by_used_features_f(
        eval_xs_, X_train.shape[-1], max_features)
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)

    x_src = model.encoder(x_all_torch.unsqueeze(1)[:len(X_train)])
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    output = model.transformer_encoder(train_x)
    linear_model_coefs = model.decoder(output)
    encoder_weight = model.encoder.get_parameter("weight")
    encoder_bias = model.encoder.get_parameter("bias")

    total_weights = torch.matmul(encoder_weight[:, :n_features].T, linear_model_coefs[0, :-1, :n_classes])
    total_biases = torch.matmul(encoder_bias, linear_model_coefs[0, :-1, :n_classes]) + linear_model_coefs[0, -1, :n_classes]
    return total_weights.detach().cpu().numpy() / (n_features / max_features), total_biases.detach().cpu().numpy()


def extract_mlp_model(model, X_train, y_train, device="cpu", inference_device="cpu", scale=True):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train).to(device)
    xs = torch.Tensor(X_train).to(device)
    if scale:
        eval_xs_ = normalize_data(xs, eval_position)
    else:
        eval_xs_ = torch.clip(xs, min=-100, max=100)

    eval_xs = normalize_by_used_features_f(
        eval_xs_, X_train.shape[-1], max_features)
    if X_train.shape[1] > 100:
        raise ValueError("Cannot run inference on data with more than 100 features")
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
        x = repeat(model.latents, 'n d -> b n d', b=data.shape[0])

        # layers
        for cross_attn, cross_ff, self_attns in model.layers:
            x = cross_attn(x, context=data) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        output = rearrange(x, 'b n d -> n b d')
    (b1, w1), *layers = model.decoder(output)

    w1_data_space_prenorm = w1.squeeze()[:n_features, :]
    b1_data_space = b1.squeeze()

    w1_data_space = w1_data_space_prenorm / (n_features / max_features)

    if model.decoder.weight_embedding_rank is not None:
        w1_data_space = torch.matmul(w1_data_space, model.decoder.shared_weights[0])

    layers_result = [(b1_data_space, w1_data_space)]

    for i, (b, w) in enumerate(layers[:-1]):
        if model.decoder.weight_embedding_rank is not None:
            w = torch.matmul(w, model.decoder.shared_weights[i + 1])
        layers_result.append((b.squeeze(), w.squeeze()))

    # remove extra classes on output layer
    layers_result.append((layers[-1][0].squeeze()[:n_classes], layers[-1][1].squeeze()[:, :n_classes]))

    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()

    return [(detach(b), detach(w)) for (b, w) in layers_result]


def predict_with_linear_model(X_train, X_test, weights, biases):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1) + .000001
    X_test_scaled = (X_test - mean) / std
    X_test_scaled = np.clip(X_test_scaled, a_min=-100, a_max=100)
    res2 = np.dot(X_test_scaled, weights) + biases
    from scipy.special import softmax
    return softmax(res2 / .8, axis=1)


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


def predict_with_mlp_model(X_train, X_test, layers, scale=True, inference_device="cpu"):
    if inference_device == "cpu":
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0, ddof=1) + .000001
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        if scale:
            std[np.isnan(std)] = 1
            X_test_scaled = (X_test - mean) / std
        else:
            X_test_scaled = X_test
        out = np.clip(X_test_scaled, a_min=-100, a_max=100)
        for i, (b, w) in enumerate(layers):
            out = np.dot(out, w) + b
            if i != len(layers) - 1:
                out = np.maximum(out, 0)
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        from scipy.special import softmax
        return softmax(out / .8, axis=1)
    elif "cuda" in inference_device:
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        if scale:
            X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        else:
            X_test_scaled = torch.Tensor(X_test).to(inference_device)
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class MotherNetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", label_offset=0, scale=True, inference_device="cpu"):
        self.path = path
        self.device = device
        self.label_offset = label_offset
        self.inference_device = inference_device
        self.scale = scale

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        model, config = load_model(self.path, device=self.device)
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] != "mlp":
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        n_classes = len(le.classes_)
        indices = np.mod(np.arange(n_classes) + self.label_offset, n_classes)
        layers = extract_mlp_model(model, X, np.mod(y + self.label_offset, n_classes), device=self.device,
                                   inference_device=self.inference_device, scale=self.scale)
        if self.label_offset == 0:
            self.parameters_ = layers
        else:
            *lower_layers, b_last, w_last = layers
            self.parameters_ = (*lower_layers, (b_last[indices], w_last[:, indices]))
        self.classes_ = le.classes_
        return self

    def predict_proba(self, X):
        return predict_with_mlp_model(self.X_train_, X, self.parameters_, scale=self.scale, inference_device=self.inference_device)

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
    def __init__(self, base_estimator, n_estimators=32, random_state=None, power=True, label_shift=True, feature_shift=True, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.power = power
        self.label_shift = label_shift
        self.feature_shift = feature_shift
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.array(X)
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        use_power_transformer = [True, False] if self.power else [False]
        feature_shifts = list(range(self.n_features_)) if self.feature_shift else [0]
        label_shifts = list(range(self.n_classes_)) if self.label_shift else [0]
        shifts = list(itertools.product(label_shifts, feature_shifts, use_power_transformer))
        rng = random.Random(self.random_state)
        shifts = rng.sample(shifts, min(len(shifts), self.n_estimators))
        estimators = []
        n_jobs = self.n_jobs if X.shape[0] > 1000 else 1

        for label_shift, feature_shift, use_power_transformer in shifts:
            estimator = ShiftClassifier(self.base_estimator, feature_shift=feature_shift, label_shift=label_shift)
            if use_power_transformer:
                # remove zero-variance features to avoid division by zero
                estimator = Pipeline([('variance_threshold', VarianceThreshold()), ('scale', StandardScaler()),
                                     ('power_transformer', PowerTransformer()), ('shift_classifier', estimator)])
            estimators.append((str((label_shift, feature_shift, use_power_transformer)), estimator))
        self.vc_ = VotingClassifier(estimators, voting='soft', n_jobs=n_jobs)
        self.vc_.fit(X, y)
        return self

    @property
    def device(self):
        return self.base_estimator.device

    def predict_proba(self, X):
        X = np.array(X)
        # numeric instabilities propagate and sklearn's metrics don't like it.
        probs = self.vc_.predict_proba(X)
        probs /= probs.sum(axis=1).reshape(-1, 1)
        return probs

    def predict(self, X):
        X = np.array(X)
        return self.vc_.predict(X)

    @property
    def classes_(self):
        return self.vc_.classes_

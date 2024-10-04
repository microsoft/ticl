import itertools
import random

import numpy as np
import torch
from einops import rearrange, repeat
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.compose import make_column_transformer

from mothernet.model_builder import load_model
from mothernet.utils import normalize_by_used_features_f, normalize_data, get_mn_model
from mothernet.evaluation.baselines.torch_mlp import TorchMLP, NeuralNetwork


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

    x_src = model.encoder(x_all_torch.unsqueeze(1))
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    output = model.transformer_encoder(train_x)
    linear_model_coefs = model.decoder(output)
    encoder_weight = model.encoder.get_parameter("weight")
    encoder_bias = model.encoder.get_parameter("bias")

    total_weights = torch.matmul(encoder_weight[:, :n_features].T, linear_model_coefs[0, :-1, :n_classes])
    total_biases = torch.matmul(encoder_bias, linear_model_coefs[0, :-1, :n_classes]) + linear_model_coefs[0, -1, :n_classes]
    return total_weights.detach().cpu().numpy() / (n_features / max_features), total_biases.detach().cpu().numpy()


def extract_mlp_model(model, config, X_train, y_train, device="cpu", inference_device="cpu", scale=True):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    try:
        max_features = config['prior']['num_features']
    except KeyError:
        max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    if torch.is_tensor(X_train):
        xs = X_train.to(device)
    else:
        xs = torch.Tensor(X_train.astype(float)).to(device)
    if torch.is_tensor(y_train):
        ys = y_train.to(device)
    else:
        ys = torch.Tensor(y_train.astype(float)).to(device)

    if scale:
        eval_xs_ = normalize_data(xs, eval_position)
    else:
        eval_xs_ = torch.clip(xs, min=-100, max=100)

    eval_xs = normalize_by_used_features_f(
        eval_xs_, X_train.shape[-1], max_features)
    if X_train.shape[1] > max_features:
        raise ValueError(f"Cannot run inference on data with more than {max_features} features")
    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], max_features - X_train.shape[1]), device=device)], axis=1)
    x_src = model.encoder(x_all_torch.unsqueeze(1))

    if model.y_encoder is not None:
        y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
        train_x = x_src + y_src
    else:
        train_x = x_src

    if hasattr(model, "transformer_encoder"):
        # tabpfn mlp model maker
        output = model.transformer_encoder(train_x)
    elif hasattr(model, "ssm"):
        # ssm model maker
        output = model.ssm(train_x)
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
    (b1, w1), *layers = model.decoder(output, ys)

    w1_data_space_prenorm = w1.squeeze()[:n_features, :]
    b1_data_space = b1.squeeze()

    w1_data_space = w1_data_space_prenorm / (n_features / max_features)

    if model.decoder.weight_embedding_rank is not None and len(layers):
        w1_data_space = torch.matmul(w1_data_space, model.decoder.shared_weights[0])

    layers_result = [(b1_data_space, w1_data_space)]

    for i, (b, w) in enumerate(layers[:-1]):
        if model.decoder.weight_embedding_rank is not None:
            w = torch.matmul(w, model.decoder.shared_weights[i + 1])
        layers_result.append((b.squeeze(), w.squeeze()))

    # remove extra classes on output layer
    if len(layers):
        layers_result.append((layers[-1][0].squeeze()[:n_classes], layers[-1][1].squeeze()[:, :n_classes]))
    else:
        layers_result = [(b1_data_space[:n_classes], w1_data_space[:, :n_classes])]

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


def predict_with_mlp_model(train_mean, train_std, X_test, layers, scale=True, inference_device="cpu", config=None):
    if inference_device == "cpu":
        X_test = np.array(X_test, dtype=float)
        # FIXME replacing nan with 0 as in TabPFN
        X_test = np.nan_to_num(X_test, 0)
        if scale:
            X_test_scaled = (X_test - train_mean) / train_std
        else:
            X_test_scaled = X_test
        out = np.clip(X_test_scaled, a_min=-100, a_max=100)
        for i, (b, w) in enumerate(layers):
            out = np.dot(out, w) + b
            if i != len(layers) - 1:
                try:
                    activation = config['mothernet']['predicted_activation']
                except (KeyError, TypeError):
                    activation = "relu"
                if activation != "relu":
                    raise ValueError(f"Only ReLU activation supported, got {activation}")
                out = np.maximum(out, 0)
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        from scipy.special import softmax
        return softmax(out / .8, axis=1)
    elif "cuda" in inference_device:
        mean = torch.Tensor(train_mean).to(inference_device)
        std = torch.Tensor(train_std).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_test = torch.Tensor(X_test).to(inference_device).nan_to_num(0)
        if scale:
            X_test_scaled = (X_test - mean) / std
        else:
            X_test_scaled = X_test
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class MotherNetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", label_offset=0, scale=True, inference_device="cpu", model=None, config=None):
        self.path = path
        self.device = device
        self.label_offset = label_offset
        self.inference_device = inference_device
        self.scale = scale
        if model is not None and path is not None:
            raise ValueError("Only one of path or model must be provided")
        if model is not None and config is None:
            raise ValueError("config must be provided if model is provided")
        self.model = model
        self.config = config

        if path is None and model is None:
            model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
            path = get_mn_model(model_string)
        self.path = path

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) > 10:
            raise ValueError(f"Only 10 classes supported, found {len(le.classes_)}")
        if self.model is not None:
            model = self.model
            config = self.config
        else:
            model, config = load_model(self.path, device=self.device)
            self.config = config
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["mlp", "mothernet", 'ssm_mothernet']:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        n_classes = len(le.classes_)
        indices = np.mod(np.arange(n_classes) + self.label_offset, n_classes)
        layers = extract_mlp_model(model, config, X, np.mod(y + self.label_offset, n_classes), device=self.device,
                                   inference_device=self.inference_device, scale=self.scale)
        if self.label_offset == 0:
            self.parameters_ = layers
        else:
            *lower_layers, b_last, w_last = layers
            self.parameters_ = (*lower_layers, (b_last[indices], w_last[:, indices]))
        self.classes_ = le.classes_
        self.mean_ = np.nan_to_num(np.nanmean(X, axis=0), 0)
        self.std_ = np.nanstd(X, axis=0, ddof=1) + .000001
        self.std_[np.isnan(self.std_)] = 1

        return self

    def predict_proba(self, X):
        return predict_with_mlp_model(self.mean_, self.std_, X, self.parameters_, scale=self.scale, inference_device=self.inference_device)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MotherNetInitMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cuda", learning_rate=1e-3, n_epochs=0, verbose=0, weight_decay=0, dropout_rate=0):
        self.path = path
        self.device = device
        if path is None:
            model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
            path = get_mn_model(model_string)
        self.path = path
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) > 10:
            raise ValueError(f"Only 10 classes supported, found {len(le.classes_)}")
        model, config = load_model(self.path, device=self.device)
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["mlp", "mothernet"]:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        layers = extract_mlp_model(model, config, X, y, device=self.device,
                                   inference_device=self.device, scale=True)
        hidden_size = config['mothernet']['predicted_hidden_layer_size']
        n_layers = config['mothernet']['predicted_hidden_layers']
        assert len(layers) == n_layers + 1  # n_layers counts number of hidden layers
        nn = NeuralNetwork(n_features=X.shape[1], n_classes=len(le.classes_), hidden_size=hidden_size, n_layers=n_layers)
        state_dict = {}
        for i, layer in enumerate(layers):
            state_dict[f"model.linear{i}.weight"] = torch.Tensor(layer[1]).T
            state_dict[f"model.linear{i}.bias"] = torch.Tensor(layer[0])
        nn.load_state_dict(state_dict)
        try:
            nonlinearity = config['mothernet']['predicted_activation']
        except (KeyError, TypeError):
            nonlinearity = "relu"
        self.mlp = TorchMLP(hidden_size=hidden_size, n_layers=n_layers, learning_rate=self.learning_rate,
                            device=self.device, n_epochs=self.n_epochs, verbose=self.verbose, init_state=nn.state_dict(),
                            nonlinearity=nonlinearity, dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)
        self.scaler = StandardScaler().fit(X)
        self.mlp.fit(X, y)
        self.parameters_ = layers
        self.classes_ = le.classes_
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict_proba(X_scaled)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


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
    def __init__(self, base_estimator, n_estimators=8, cat_features=None, random_state=None, power=True,
                 label_shift=True, feature_shift=True, onehot=True, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.power = power  # now using quantile transformer, not power transformer, but keeping the name
        self.label_shift = label_shift
        self.feature_shift = feature_shift
        self.n_jobs = n_jobs
        self.cat_features = cat_features
        self.onehot = onehot

    def fit(self, X, y):
        X = np.array(X)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.power:
            use_power_transformer = [True, False]
        else:
            [False]
        use_onehot = [True, False] if self.onehot and self.cat_features is not None and len(self.cat_features) else [False]
        feature_shifts = list(range(self.n_features_)) if self.feature_shift else [0]
        label_shifts = list(range(self.n_classes_)) if self.label_shift else [0]
        shifts = list(itertools.product(label_shifts, feature_shifts, use_power_transformer, use_onehot))
        rng = random.Random(self.random_state)
        shifts = rng.sample(shifts, min(len(shifts), self.n_estimators))
        estimators = []

        for label_shift, feature_shift, power_transformer, onehot in shifts:
            clf = ShiftClassifier(self.base_estimator, feature_shift=feature_shift, label_shift=label_shift)
            estimators.append((power_transformer, onehot, clf))

        if self.cat_features is not None and len(self.cat_features):
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[self.cat_features] = True
            self.cat_mask_ = mask
            X_cat = X[:, mask]
            X_cont = X[:, ~mask]
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', max_categories=10, sparse_output=False)
            X_cat_ohe = self.ohe_.fit_transform(X_cat)
        else:
            X_cont = X
            X_cat = None

        if X_cont.shape[1] > 0:
            self.quantile_ = QuantileTransformer()
            X_cont_quantile = self.quantile_.fit_transform(X_cont)

        self.estimators_ = []
        for power_transformer, onehot, clf in estimators:
            if X_cont.shape[1] == 0:
                X_preprocessed = X_cat_ohe
            else:
                if power_transformer:
                    X_cont_preprocessed = X_cont_quantile
                else:
                    X_cont_preprocessed = X_cont
                if onehot:
                    X_preprocessed = np.concatenate([X_cat_ohe, X_cont_preprocessed], axis=1)
                elif X_cat is not None:
                    X_preprocessed = np.concatenate([X_cat, X_cont_preprocessed], axis=1)
                else:
                    X_preprocessed = X_cont_preprocessed
            skb = None
            if X_preprocessed.shape[1] > 100:
                skb = SelectKBest(k=100)
                X_preprocessed = skb.fit_transform(np.nan_to_num(X_preprocessed, 0), y)
            clf.fit(X_preprocessed, y)
            self.estimators_.append((power_transformer, onehot, skb, clf))

        return self

    @property
    def device(self):
        return self.base_estimator.device

    def predict_proba(self, X):
        X = np.array(X)
        predicted_probas = []
        if self.cat_features is not None and len(self.cat_features):
            X_cat = X[:, self.cat_mask_]
            X_cont = X[:, ~self.cat_mask_]
            X_cat_ohe = self.ohe_.transform(X_cat)
        else:
            X_cont = X
            X_cat = None

        if X_cont.shape[1] > 0:
            X_cont_quantile = self.quantile_.transform(X_cont)

        for power_transformer, onehot, skb, clf in self.estimators_:
            if X_cont.shape[1] == 0:
                X_preprocessed = X_cat_ohe
            else:
                if power_transformer:
                    X_cont_preprocessed = X_cont_quantile
                else:
                    X_cont_preprocessed = X_cont
                if onehot:
                    X_preprocessed = np.concatenate([X_cat_ohe, X_cont_preprocessed], axis=1)
                elif X_cat is not None:
                    X_preprocessed = np.concatenate([X_cat, X_cont_preprocessed], axis=1)
                else:
                    X_preprocessed = X_cont_preprocessed
            if X_preprocessed.shape[1] > 100:
                X_preprocessed = skb.transform(np.nan_to_num(X_preprocessed, 0))
            predicted_probas.append(clf.predict_proba(X_preprocessed))
        return np.mean(predicted_probas, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
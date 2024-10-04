from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ticl.model_builder import load_model
from ticl.models.biattention_additive_mothernet import _determine_is_categorical
from ticl.models.encoders import get_fourier_features
from ticl.models.utils import bin_data
from ticl.utils import normalize_data

from interpret.glassbox._ebm._ebm import EBMExplanation
from interpret.utils._explanation import gen_global_selector


def extract_additive_model(model, X_train, y_train, config=None, device="cpu", inference_device="cpu", regression=False, is_categorical: List[List] = None):
    try:
        max_features = config['prior']['num_features']
    except KeyError:
        max_features = 100

    try:
        pad_zeros = config['prior']['classification']['pad_zeros']
    except KeyError:
        pad_zeros = True

    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    with torch.no_grad():
        n_classes = 1 if regression else len(np.unique(y_train))
        n_features = X_train.shape[1]

        ys = torch.Tensor(y_train).to(device)
        xs = torch.Tensor(X_train).to(device)

        if pad_zeros:
            x_all_torch = torch.concat([xs, torch.zeros((X_train.shape[0], max_features - X_train.shape[1]), device=device)], axis=1)
        else:
            x_all_torch = xs

        X_onehot, bin_edges = bin_data(x_all_torch.unsqueeze(1), n_bins=model.n_bins, nan_bin=model.nan_bin,
                                       sklearn_binning=model.sklearn_binning)
        X_onehot, bin_edges = X_onehot.squeeze(1), bin_edges.squeeze(1)

        if model.input_layer_norm:
            X_onehot = model.input_norm(X_onehot.float())
        if getattr(model, "fourier_features", 0) > 0:
            x_scaled = normalize_data(xs)
            x_fourier = get_fourier_features(x_scaled, model.fourier_features)
            X_onehot = torch.cat([X_onehot, x_fourier], -1)

        x_src = model.encoder(X_onehot.unsqueeze(1).float())
        if getattr(model, 'categorical_embedding', False):
            is_categorical = _determine_is_categorical(x_src, info={'categorical_features': is_categorical})
            x_src = x_src + model.categorical_embedding(is_categorical, inference=True)

        if model.y_encoder is None:
            train_x = x_src
        else:
            y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
            if x_src.ndim == 4:
                # baam model, per feature
                y_src = y_src.unsqueeze(-2)
            train_x = x_src + y_src
        assert train_x.shape == x_src.shape
        if hasattr(model, "layers"):
            # baam model
            output = train_x
            for mod in model.layers:
                output = mod(output)
        else:
            output = model.transformer_encoder(train_x)

        if model.marginal_residual in [True, 'True', 'output', 'decoder']:
            class_averages = model.class_average_layer(X_onehot.float().unsqueeze(1), ys.unsqueeze(1))
            # class averages are batch x outputs x features x bins
            # output is batch x features x bins x outputs
            marginals = model.marginal_residual_layer(class_averages)

        if model.marginal_residual == 'decoder':
            weights, biases = model.decoder(output, ys, marginals)
        else:
            weights, biases = model.decoder(output, ys)

        if model.marginal_residual in [True, 'True', 'output', 'decoder']:
            if hasattr(model, "layers") and len(model.layers) == 0:
                weights = marginals.permute(0, 2, 3, 1)
            else:
                weights = weights + marginals.permute(0, 2, 3, 1)
        w = weights.squeeze(0)[:n_features, :, :n_classes]

        if biases is None:
            b = torch.zeros(n_classes, device=device)
        else:
            b = biases.squeeze()[:n_classes]
        bins_data_space = bin_edges[:n_features]

    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()

    return detach(w), detach(b), detach(bins_data_space)


def predict_with_additive_model(X_train, X_test, weights, biases, bin_edges, nan_bin, inference_device="cpu",
                                regression=False):
    additive_components = []
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[1] == len(weights)
    assert weights.shape[0] == X_train.shape[1]
    n_bins = weights.shape[1]
    assert bin_edges.shape == (X_train.shape[1], n_bins - 1 - int(nan_bin))
    if inference_device == "cpu":
        out = np.zeros((X_test.shape[0], weights.shape[-1]))
        for col, bins, w in zip(X_test.T, bin_edges, weights):
            binned = np.searchsorted(bins, col)
            if nan_bin:
                # Put NaN data on the last bin.
                binned[np.isnan(col)] = n_bins - 1
            out += w[binned]
            additive_components.append(w[binned])
        out += biases
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        if regression:
            return out, additive_components
        from scipy.special import softmax
        return softmax(out / .8, axis=1), additive_components
    elif "cuda" in inference_device:
        raise NotImplementedError('CUDA inference not working')
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        import pdb
        pdb.set_trace()
        raise NotImplementedError
        layers = None
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")
 

class ExplainableAdditivePredictor:
    def explain_global(self):
        # Start creating properties in the same style as EBM to leverage existing explanations

        # Loop over features to extract term_scores_

        self.term_scores_ = []
        for feature_idx in range(self.w_.shape[0]):
            if self.w_.shape[2] == 2:  # binary classification
                class_one_scores = self.w_[feature_idx, :, 1] - self.w_[feature_idx, :, 0]
                padded_scores = np.pad(class_one_scores, (1, 1), 'constant', constant_values=(0, 0))
            elif self.w_.shape[2] == 1:  # regression
                class_one_scores = self.w_[feature_idx, :]
                padded_scores = np.pad(class_one_scores, (1, 1), 'constant', constant_values=(0, 0))
            else:
                raise Exception("Need to implement explanations for multiclass")
            
            self.term_scores_.append(padded_scores)
        
        lower_bound, upper_bound = np.inf, -np.inf
        for scores in self.term_scores_:
            lower_bound = min(lower_bound, np.min(scores))
            upper_bound = max(upper_bound, np.max(scores))

        bounds = (lower_bound, upper_bound)

        # TODO: Update to include real feature names
        term_names = [f"Feature {i}" for i in range(self.w_.shape[0])]
        term_types = ["continuous"] * len(term_names) # TODO: Currently assume all numeric features

        data_dicts = []
        feature_list = []
        density_list = []

        # loop over features
        for i in range(self.w_.shape[0]):
            model_graph = self.term_scores_[i]
            errors = None
            feature_bins = self.bin_edges_[i]

            min_graph = np.nan
            max_graph = np.nan
            feature_bounds = getattr(self, "feature_bounds_", None)
            if feature_bounds is not None:
                min_graph = feature_bounds[i][0]
                max_graph = feature_bounds[i][1]

            bin_labels = list(
                np.concatenate(([min_graph], feature_bins, [max_graph]))
            )

            scores = list(model_graph)
            density_dict = {
                "names": None,
                "scores": None,
            }
            density_list.append(density_dict)

            data_dict = {
                "type": "univariate",
                "names": bin_labels,
                "scores": np.array(scores)[1:-1],
                "scores_range": bounds,
                "upper_bounds": None if errors is None else model_graph + errors,
                "lower_bounds": None if errors is None else model_graph - errors,
                # "density": {
                #     "names": names,
                #     "scores": densities,
                # },
            }
            if hasattr(self, "classes_"):
                # Classes should be numpy array, convert to list.
                data_dict["meta"] = {"label_names": self.classes_.tolist()}

            data_dicts.append(data_dict)

        overall_dict = {
            "type": "univariate",
            "names": term_names,
            "scores": [1 for i in range(len(term_names))], # TODO: Stop hard coding
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=term_names,
            feature_types=term_types,
            name="Mothernet Explanation",
            selector=gen_global_selector(
                len(term_names),
                term_names,
                term_types,
                getattr(self, "unique_val_counts_", None),
                None,
            ),
        )
    
    def _extract_feature_bounds(self, X):
        if torch.is_tensor(X):
            X_numpy = X.cpu().detach().numpy()
        else:
            X_numpy = X
        mins = X_numpy.min(axis=0).tolist()
        maxs = X_numpy.max(axis=0).tolist()
        feature_bounds = [(float(min_), float(max_)) for min_, max_ in zip(mins, maxs)]
        
        self.feature_bounds_ = feature_bounds
        return self
    

class MotherNetAdditiveClassifier(ClassifierMixin, BaseEstimator, ExplainableAdditivePredictor):
    def __init__(self, path=None, device="cpu", inference_device="cpu", model=None, config=None, cat_features: List[int] = None):
        self.path = path
        self.device = device
        self.inference_device = inference_device
        if model is None and path is None:
            raise ValueError("Either path or model must be provided")
        if model is not None and path is not None:
            raise ValueError("Only one of path or model must be provided")
        if model is not None and config is None:
            raise ValueError("config must be provided if model is provided")
        self.model = model
        self.config = config
        if hasattr(model, "nan_bin"):
            self.nan_bin = model.nan_bin
        else:
            self.nan_bin = False
        if hasattr(model, "sklearn_binning"):
            self.sklearn_binning = model.sklearn_binning
        else:
            self.sklearn_binning = False
        self.cat_features = cat_features

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        if self.model is not None:
            model, config = self.model, self.config
        else:
            model, config = load_model(self.path, device=self.device)
            self.config_ = config
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["additive", "baam"]:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
    
        try:
            self.nan_bin = config['additive']['nan_bin']
        except KeyError:
            self.nan_bin = False

        w, b, bin_edges = extract_additive_model(model, X, y, config=config, device=self.device, inference_device=self.inference_device,
                                                 is_categorical=self.cat_features, regression=False)

        self.w_ = w
        self.b_ = b
        self.bin_edges_ = bin_edges
        self.classes_ = le.classes_
        self._extract_feature_bounds(X)

        return self

    def predict_proba(self, X):
        p, components = self.predict_proba_with_additive_components(X)
        return p

    def predict_proba_with_additive_components(self, X):
        return predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, nan_bin=self.nan_bin,
                                           inference_device=self.inference_device, regression=False)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MotherNetAdditiveRegressor(RegressorMixin, BaseEstimator, ExplainableAdditivePredictor):
    def __init__(self, path=None, device="cpu", inference_device="cpu", model=None, config=None, cat_features: List[int] = None):
        self.path = path
        self.device = device
        self.inference_device = inference_device
        if model is None and path is None:
            raise ValueError("Either path or model must be provided")
        if model is not None and path is not None:
            raise ValueError("Only one of path or model must be provided")
        if model is not None and config is None:
            raise ValueError("config must be provided if model is provided")
        self.model = model
        self.config = config
        self.cat_features = cat_features

    def fit(self, X, y):
        self.X_train_ = X
        if self.model is not None:
            model, config = self.model, self.config
        else:
            model, config = load_model(self.path, device=self.device)
            self.config_ = config
        if config['model_type'] not in ["additive", "baam"]:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        self.nan_bin = config['additive']['nan_bin']
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        w, b, bin_edges = extract_additive_model(model, X, y_scaled, config=config, device=self.device, inference_device=self.inference_device,
                                                 regression=True, is_categorical=self.cat_features)
        self.w_ = w
        self.b_ = b
        self.bin_edges_ = bin_edges
        self._extract_feature_bounds(X)

    def predict(self, X):
        y_pred = predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, nan_bin=self.nan_bin,
                                             inference_device=self.inference_device, regression=True)[0]
        return self._y_scaler.inverse_transform(y_pred).ravel()

    def predict_with_additive_components(self, X):
        y_pred, components = predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, nan_bin=self.nan_bin,
                                                         inference_device=self.inference_device, regression=True)
        return self._y_scaler.inverse_transform(y_pred).ravel(), components

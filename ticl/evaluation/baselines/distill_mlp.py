import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ticl.prediction.tabpfn import TabPFNClassifier
from ticl.evaluation.baselines.torch_mlp import TorchMLP, _encode_y


class DistilledTabPFNMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, n_epochs=10, hidden_size=128, n_layers=2, learning_rate=1e-3, device="cpu", dropout_rate=0.0, layernorm=False, categorical_features=None, N_ensemble_configurations=32, verbose=0, **kwargs):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm
        self.categorical_features = categorical_features
        self.N_ensemble_configurations = N_ensemble_configurations
        self.kwargs = kwargs
        self.verbose = verbose

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        tbfn = TabPFNClassifier(N_ensemble_configurations=self.N_ensemble_configurations,
                                device=self.device, verbose=self.verbose, **self.kwargs).fit(X, y)
        y_train_soft_probs = tbfn.predict_proba(X)
        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                             n_layers=self.n_layers, dropout_rate=self.dropout_rate, device=self.device, layernorm=self.layernorm, verbose=self.verbose)

        self.mlp_.fit(X, y_train_soft_probs)
        return self

    def predict(self, X):
        return self.classes_[self.mlp_.predict(X)]

    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
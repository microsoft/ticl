from sklearn.base import ClassifierMixin, BaseEstimator
import torch
from torch.utils.data import TensorDataset

import torch
import numpy as np
from collections import OrderedDict
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from warnings import filterwarnings, catch_warnings

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

class NeuralNetwork(nn.Module):
    def __init__(self, n_features=784, n_classes=10, hidden_size=128, n_layers=2, dropout_rate=0.0, layernorm=False):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm
        # create a list of linear and activation layers
        layers = [nn.Linear(n_features, hidden_size), nn.ReLU()]
        # add more hidden layers with optional dropout
        for _ in range(n_layers - 1):
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            if layernorm:
                layers.append(nn.LayerNorm(hidden_size))
        # add the output layer
        layers.append(nn.Linear(hidden_size, n_classes))
        # create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # pass the input through the model
        return self.model(x)
    
def _encode_y(y):        
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    if y.ndim == 1:
        le = LabelEncoder()
        y = le.fit_transform(y)
        classes = le.classes_
    else:
        # used probabilities as labels
        classes = torch.arange(y.shape[1])
    return y, classes

class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_size=128, n_epochs=10, learning_rate=1e-3, n_layers=2,
                 verbose=0, dropout_rate=0.0, device='cuda', layernorm=False, weight_decay=0.01, batch_size=None):
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.device = device
        self.layernorm = layernorm
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def fit_from_dataloader(self, dataloader, n_features, classes):
        model = NeuralNetwork(n_features=n_features, n_classes=len(classes), n_layers=self.n_layers,
                              hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, layernorm=self.layernorm)
        model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.n_epochs):
            size = len(dataloader.dataset)
            losses = []
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)
                losses.append(loss.item())
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 and self.verbose:
                loss, current = np.mean(losses), (batch + 1) * len(X)
                print(f"epoch: {epoch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.model_ = model
        self.classes_ = classes

    def fit(self, X, y):
        y, classes = _encode_y(y)
        if torch.is_tensor(X):
            X = X.clone().detach().to(self.device).float()
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if torch.is_tensor(y):
            y = y.clone().detach().to(self.device)
        else:
            y = torch.tensor(y, device=self.device)
        X = X.nan_to_num()
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size or X.shape[0])
        self.fit_from_dataloader(dataloader, n_features=X.shape[1], classes=classes)
        return self

    def _predict(self, X):
        if torch.is_tensor(X):
            X = X.clone().detach().to(self.device).float()
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model_(X.nan_to_num())
        
    def predict(self, X):
        pred = self._predict(X)
        return self.classes_[pred.argmax(1).detach().cpu().numpy()]
    
    def predict_proba(self, X):
        pred = self._predict(X)
        return pred.softmax(dim=1).detach().cpu().numpy()


class DistilledTabPFNMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, temperature=1, n_epochs=10, hidden_size=128, n_layers=2, learning_rate=1e-3, device="cpu", dropout_rate=0.0, layernorm=False, categorical_features=None, N_ensemble_configurations=32, verbose=0, **kwargs):
        self.temperature = temperature
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
        self.verbose=verbose

    def fit(self, X, y):
        tbfn = TabPFNClassifier(N_ensemble_configurations=self.N_ensemble_configurations, temperature=self.temperature, device=self.device, verbose=self.verbose, **self.kwargs).fit(X, y)
        y_train_soft_probs = tbfn.predict_proba(X) * self.temperature ** 2
        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                             n_layers=self.n_layers, dropout_rate=self.dropout_rate, device=self.device, layernorm=self.layernorm, verbose=self.verbose)
	
        self.mlp_.fit(X, y_train_soft_probs)
        return self
    def predict(self, X):
        return self.mlp_.predict(X)
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
    @property
    def classes_(self):
        return self.mlp_.classes_


class DistilledMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, clf, temperature=1, n_epochs=10, hidden_size=128, n_layers=2, learning_rate=1e-3, device="cpu", dropout_rate=0.0, layernorm=False, verbose=0, **kwargs):
        self.clf = clf
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm
        self.verbose = verbose

    def fit(self, X, y):
        y, self.classes_ = _encode_y(y)
        y_train_soft_probs = self.clf.predict_proba(X) * self.temperature ** 2
        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                             n_layers=self.n_layers, dropout_rate=self.dropout_rate, device=self.device, layernorm=self.layernorm, verbose=self.verbose)
	
        self.mlp_.fit(X, y_train_soft_probs)
        return self
    def predict(self, X):
        return self.classes_[self.mlp_.predict(X)]
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)


class SmoteAugmentedDataset(IterableDataset):
    def __init__(self, X, y, tabpfn, categorical_features=None, upsample_rate=2, temperature=1, device='cpu'):
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        self.X = X
        self.y = y
        self.tabpfn = tabpfn
        self.categorical_features = categorical_features
        self.upsample_rate = upsample_rate
        self.random_state = np.random.RandomState(42)
        self.new_counts = (pd.value_counts(y) * self.upsample_rate).to_dict()
        self.temperature = temperature
        self.device = device

    def __len__(self):
        return len(self.X) * self.upsample_rate
    
    def __iter__(self):
        random_seed = self.random_state.randint(0, 2**32-1)
        from imblearn.over_sampling import SMOTENC, SMOTE
        if self.categorical_features is None:
            smote = SMOTE(sampling_strategy=self.new_counts, random_state=random_seed)
        else:
            smote = SMOTENC(sampling_strategy=self.new_counts, categorical_features=self.categorical_features, random_state=random_seed)
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            X_new, _ = smote.fit_resample(self.X, self.y)
        y_train_soft_probs = self.tabpfn.predict_proba(X_new) * self.temperature ** 2
        X_return = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        y_return = torch.tensor(y_train_soft_probs, device=self.device)
        yield X_return, y_return


class DistilledTabPFNMLPUpsampler(ClassifierMixin, BaseEstimator):
    def __init__(self, temperature=1, n_epochs=10, hidden_size=128, n_layers=2, learning_rate=1e-3, device="cpu", dropout_rate=0.0, layernorm=False, upsample_rate=2, categorical_features=None, N_ensemble_configurations=32, **kwargs):
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm
        self.upsample_rate = upsample_rate
        self.categorical_features = categorical_features
        self.N_ensemble_configurations = N_ensemble_configurations
        self.kwargs = kwargs

    def fit(self, X, y):
        tabpfn = TabPFNClassifier(N_ensemble_configurations=self.N_ensemble_configurations, temperature=self.temperature, device=self.device, **self.kwargs).fit(X, y, overwrite_warning=True)
        augmented_dataset = SmoteAugmentedDataset(X, y, categorical_features=self.categorical_features, upsample_rate=self.upsample_rate, tabpfn=tabpfn, device=self.device, temperature=self.temperature)

        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                             n_layers=self.n_layers, dropout_rate=self.dropout_rate, device=self.device, layernorm=self.layernorm)
        dataloader = DataLoader(augmented_dataset, batch_size=None)
        # classes is range since we always have y as probabilities
        self.mlp_.fit_from_dataloader(dataloader, n_features=X.shape[1], classes=np.arange(len(tabpfn.classes_)))
        return self
    def predict(self, X):
        return self.mlp_.predict(X)
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
    @property
    def classes_(self):
        return self.mlp_.classes_


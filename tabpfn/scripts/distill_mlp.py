from sklearn.base import ClassifierMixin, BaseEstimator
import torch
from torch.utils.data import TensorDataset

import torch
import numpy as np
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

class NeuralNetwork(nn.Module):
    def __init__(self, n_features=784, n_classes=10, hidden_size=512, n_layers=2, dropout_rate=0.0, layernorm=False):
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

class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_size=512, n_epochs=10, learning_rate=1e-3, n_layers=2,
                 verbose=0, dropout_rate=0.0, device='cuda', layernorm=False):
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.device = device
        self.layernorm = layernorm

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        if y.ndim == 1:
            self.classes_ = np.unique(y)
            y = self.le_.fit_transform(y)
            self.classes_ = self.le_.classes_
        else:
            self.classes_ = torch.arange(y.shape[1])

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, device=self.device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=X.shape[0])
        model = NeuralNetwork(n_features=X.shape[1], n_classes=len(self.classes_), n_layers=self.n_layers,
                              hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, layernorm=self.layernorm)
        model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.n_epochs):
            size = len(dataloader.dataset)
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0 and self.verbose:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.model_ = model
        return self
        
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        pred = self.model_(X)
        return self.classes_[pred.argmax(1).detach().cpu().numpy()]
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        pred = self.model_(X)
        return pred.softmax(dim=1).detach().cpu().numpy()


class DistilledTabPFNMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, temperature=1, n_epochs=10, hidden_size=512, n_layers=2, learning_rate=1e-3, device="cpu", dropout_rate=0.0, layernorm=False):
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm

    def fit(self, X, y):
        tbfn = TabPFNClassifier(N_ensemble_configurations=32, temperature=self.temperature, device=self.device).fit(X, y)
        y_train_soft_probs = tbfn.predict_proba(X) * self.temperature ** 2
        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size,
                             n_layers=self.n_layers, dropout_rate=self.dropout_rate, device=self.device, layernorm=self.layernorm)
        self.mlp_.fit(X, y_train_soft_probs)
        return self
    def predict(self, X):
        return self.mlp_.predict(X)
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
    @property
    def classes_(self):
        return self.mlp_.classes_

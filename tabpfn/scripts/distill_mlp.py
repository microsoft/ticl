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
    def __init__(self, n_features=784, n_classes=10, hidden_size=512, n_layers=2):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        layers = OrderedDict()
        layers['input'] = nn.Linear(n_features, hidden_size)
        for i in range(1, n_layers):
            layers[f"hidden_{i - 1}_activation"] = nn.ReLU()
            layers[f"hidden_{i}"] = nn.Linear(hidden_size, hidden_size)
        layers[f"hidden_{n_layers}_activation"] = nn.ReLU()
        layers['output'] = nn.Linear(hidden_size, n_classes)
        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_size=512, n_epochs=10, learning_rate=1e-3, n_layers=2, verbose=0):
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_layers = n_layers
        
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
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        train_dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), y)
        dataloader = DataLoader(train_dataset, batch_size=X.shape[0])
        model = NeuralNetwork(n_features=X.shape[1], n_classes=len(self.classes_), n_layers=self.n_layers, hidden_size=self.hidden_size)
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
        pred = self.model_(torch.from_numpy(X.astype(np.float32)))
        return self.classes_[pred.argmax(1).detach().numpy()]
    
    def predict_proba(self, X):
        pred = self.model_(torch.from_numpy(X.astype(np.float32)))
        return pred.softmax(dim=1).detach().numpy()


class DistilledTabPFNMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, temperature=1, n_epochs=10, hidden_size=512, n_layers=2, learning_rate=1e-3, device="cpu"):
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.n_layers = n_layers
    def fit(self, X, y):
        tbfn = TabPFNClassifier(N_ensemble_configurations=32, temperature=self.temperature, device=self.device).fit(X, y)
        y_train_soft_probs = tbfn.predict_proba(X) * self.temperature ** 2
        self.mlp_ = TorchMLP(n_epochs=self.n_epochs, learning_rate=self.learning_rate, hidden_size=self.hidden_size, n_layers=self.n_layers).fit(X, y_train_soft_probs)
        return self
    def predict(self, X):
        return self.mlp_.predict(X)
    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)
    @property
    def classes_(self):
        return self.mlp_.classes_

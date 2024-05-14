import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from collections import OrderedDict
from abc import abstractmethod


class NeuralNetwork(nn.Module):
    def __init__(self, n_features=784, n_classes=10, hidden_size=128, n_layers=2, dropout_rate=0.0, layernorm=False, nonlinearity='relu'):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.layernorm = layernorm
        self.nonlinearity = nonlinearity
        if self.nonlinearity == 'tanh':
            nl = nn.Tanh
        elif self.nonlinearity == 'relu':
            nl = nn.ReLU
        else:
            raise ValueError(f"Unknown nonlinearity {self.nonlinearity}")
        # create a list of linear and activation layers
        layers = OrderedDict()
        layers['linear0'] = nn.Linear(n_features, hidden_size)
        layers['activation0'] = nl()
        # add more hidden layers with optional dropout

        for i in range(1, n_layers):
            if dropout_rate > 0:
                layers[f'dropout{i-1}'] = nn.Dropout(dropout_rate)
            layers[f'linear{i}'] = nn.Linear(hidden_size, hidden_size)
            layers[f'activation{i}'] = nl()
            if layernorm:
                layers[f'norm{i}'] = nn.LayerNorm(hidden_size)
        # add the output layer
        layers[f'linear{n_layers}'] = nn.Linear(hidden_size, n_classes)
        # create a sequential model
        self.model = nn.Sequential(layers)

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


class TorchModelTrainer(ClassifierMixin, BaseEstimator):
    def __init__(self, n_epochs=10, learning_rate=1e-3,
                 verbose=0, device='cuda',  weight_decay=0.01, batch_size=None, epoch_callback=None,
                 nonlinearity='relu', init_state=None):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epoch_callback = epoch_callback
        self.nonlinearity = nonlinearity
        self.init_state = init_state

    def fit_from_dataloader(self, dataloader, n_features, classes):
        model = self.make_model(n_features, len(classes))
        # loading the state dict seems the easiest way to ensure all the configs actually match
        if self.init_state is not None:
            model.load_state_dict(self.init_state)
        model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        try:
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
                if self.epoch_callback is not None:
                    self.epoch_callback(model, epoch, np.mean(losses))
        except KeyboardInterrupt:
            pass
        self.model_ = model
        self.classes_ = classes

    @abstractmethod
    def make_model(self, n_features, n_classes):
        pass

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


class TorchMLP(TorchModelTrainer):
    def __init__(self, hidden_size=128, n_epochs=10, learning_rate=1e-3, n_layers=2,
                 verbose=0, dropout_rate=0.0, device='cuda', layernorm=False, weight_decay=0.01, batch_size=None, epoch_callback=None,
                 nonlinearity='relu', init_state=None):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layernorm = layernorm
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        super().__init__(n_epochs=n_epochs, learning_rate=learning_rate, verbose=verbose,
                         device=device, init_state=init_state, batch_size=batch_size,
                         epoch_callback=epoch_callback, weight_decay=weight_decay)

    def make_model(self, n_features, n_classes):
        return NeuralNetwork(n_features=n_features, n_classes=n_classes, n_layers=self.n_layers,
                             hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, layernorm=self.layernorm,
                             nonlinearity=self.nonlinearity)
    
import logging
import time
import torch

from syne_tune import Reporter
from argparse import ArgumentParser

from tabpfn.scripts.distill_mlp import TorchMLP
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import pmlb

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--n-layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--dropout-rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--onehot', type=bool)
    parser.add_argument('--st_checkpoint_dir', type=str, default=None)

    args = parser.parse_args()
    report = Reporter()

    #x, y = np.c_[np.meshgrid(np.arange(10), np.arange(10))]
    #x, y = x.ravel(), y.ravel()

    #z = (x + y) % 7

    device = "cpu"
    torch.set_num_threads(1)

    #labels = z
    #data = np.c_[x, y]
    from sklearn.datasets import fetch_openml
    df = fetch_openml("teachingAssistant").frame
    labels = LabelEncoder().fit_transform(df['class'])
    data = df[['ID']].astype(int)

    if args.onehot:
        data = OneHotEncoder(sparse=False).fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)
    X_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, device=device)

    def epoch_callback(model, epoch, loss):
        with torch.no_grad():
            pred = model(X_test_tensor)
            acc = (pred.argmax(axis=1) == y_test_tensor).to(torch.float32).mean().item()
            print(acc)
        report(epoch=epoch + 1, accuracy=acc)

    mlp = TorchMLP(hidden_size=args.hidden_size, device=device, n_epochs=args.epochs, n_layers=args.n_layers, learning_rate=args.learning_rate,
                   dropout_rate=args.dropout_rate, weight_decay=args.weight_decay, epoch_callback=epoch_callback)
    mlp.fit(X_train, y_train)
    
    print(mlp.score(X_test, y_test))
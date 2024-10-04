import torch
from ticl.models.utils import bin_data


def test_bin_data():
    n_samples = 256
    n_bins = 64
    n_features = 4
    X = torch.randn(n_samples, 1, n_features)
    X_binned, bin_edges = bin_data(X, n_bins=n_bins)
    # check shape
    assert X_binned.shape[-1] == n_bins
    assert X_binned.shape[:-1] == X.shape
    # one bin active per feature
    assert (X_binned.sum(axis=-1) == 1).all()
    # made everything nicely divisible by n_bins so we get round numbers
    # and equal numbers of samples in each bin
    assert (X_binned.sum(axis=0).sum(axis=1) == n_samples * n_features / n_bins).all()


def test_bin_data_less_samples_than_features():
    n_samples = 3
    n_bins = 64
    n_features = 4
    X = torch.randn(n_samples, 1, n_features)
    X_binned, bin_edges = bin_data(X, n_bins=n_bins, sklearn_binning=True)
    # check shape
    assert X_binned.shape[-1] == n_bins
    assert X_binned.shape[:-1] == X.shape
    # one bin active per feature
    assert (X_binned.sum(axis=-1) == 1).all()
    # made everything nicely divisible by n_bins, so we get round numbers
    assert X_binned.sum(axis=0).sum(axis=1).to(torch.float).mean() == n_samples * n_features / n_bins

import torch
import torch.nn as nn


def bin_data(data, n_bins, nan_bin=False, single_eval_pos=None):
    if nan_bin:
        # data is samples x batch x features
        quantiles = torch.arange(n_bins, device=data.device) / (n_bins - 1)

        # Compute quantiles without nan data
        if single_eval_pos is None:
            bin_edges = torch.nanquantile(data, quantiles[1:-1], dim=0)
        else:
            bin_edges = torch.nanquantile(data[:single_eval_pos], quantiles[1:-1], dim=0)

        bin_edges = bin_edges.transpose(0, -1).contiguous()
        data = data.transpose(0, -1).contiguous()

        # Keep track of the nan positions in the data
        isnan = torch.isnan(data)

        # Fill NaNs in order to bin the data.
        data = torch.nan_to_num(data, nan=0.0)
        X_binned = torch.searchsorted(bin_edges, data)

        # Put NaN data on the last bin.
        X_binned[isnan] = n_bins - 1
        X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    else:
        # data is samples x batch x features
        data_nona = torch.nan_to_num(data, nan=0)
        quantiles = torch.arange(n_bins + 1, device=data.device) / n_bins
        if single_eval_pos is None:
            bin_edges = torch.quantile(data_nona, quantiles[1:-1], dim=0)
        else:
            bin_edges = torch.quantile(data_nona[:single_eval_pos], quantiles[1:-1], dim=0)
        zero_padding = (data_nona == 0).all(axis=0)
        # FIXME extra data copy
        bin_edges = bin_edges.transpose(0, -1).contiguous()
        data_nona = data_nona.transpose(0, -1).contiguous()
        X_binned = torch.searchsorted(bin_edges, data_nona)
        X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
        # mask zero padding data
        X_onehot[:, zero_padding, :] = 0
    return X_onehot, bin_edges

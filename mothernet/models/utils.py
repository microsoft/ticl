import torch


def bin_data(data, n_bins, single_eval_pos=None):
    # data is samples x batch x features
    # FIXME treat NaN as separate bin
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
    X_onehot = torch.nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    # mask zero padding data
    X_onehot[:, zero_padding, :] = 0
    return X_onehot, bin_edges
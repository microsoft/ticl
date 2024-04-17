import matplotlib.pyplot as plt
import numpy as np


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray, feature_names=None, feature_subset=None):
    num_classes = w.shape[2]
    num_features = len(feature_subset) if feature_subset is not None else len(bin_edges)
    if num_classes > 2:
        class_range = range(num_classes)
        rows, columns = num_classes, num_features
    else:
        class_range = [1]
        columns = min(int(np.ceil(np.sqrt(num_features))), 6)
        rows = int(np.ceil(num_features / columns))
    fig, axs = plt.subplots(rows, columns, figsize=(4*columns, 2*rows),
                            sharey=True)
    feature_range = feature_subset if feature_subset is not None else range(num_features)
    for col_idx, class_idx in enumerate(class_range):
        for ax_idx, feature_idx in enumerate(feature_range):
            if num_classes > 2:
                ax = axs[class_idx][ax_idx]
            else:
                ax = axs.ravel()[ax_idx]
            if class_idx == 0 or num_classes == 2:
                ax = axs.ravel()[feature_idx]
            bin_edge = np.concatenate(
                [[bin_edges[feature_idx][0] - (bin_edges[feature_idx][1] - bin_edges[feature_idx][0])],
                 bin_edges[feature_idx]])
            weights_normalized = w[feature_idx][:, class_idx] - w[feature_idx].mean(axis=-1)
            ax.step(bin_edge, weights_normalized, where='pre')
            if col_idx == 0:
                if feature_names is None:
                    ax.set_title(f'Feature {feature_idx}')
                else:
                    ax.set_title(f'{feature_idx}: {feature_names[feature_idx]}')

            if feature_idx == 0:
                ax.set_ylabel(f'Class {class_idx}')
    if num_classes == 2:
        for i in range(num_features, len(axs.ravel())):
            axs.ravel()[i].set_axis_off()
    plt.tight_layout()
    return axs

import matplotlib.pyplot as plt
import numpy as np


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray, feature_names=None):
    num_classes = w.shape[2]
    num_features = len(bin_edges)
    if num_classes > 2:
        class_range = range(num_classes)
        rows, columns = num_classes, num_features
    else:
        class_range = [0]
        columns = min(int(np.sqrt(num_features)), 6)
        rows = int(np.ceil(num_features / columns))
    fig, axs = plt.subplots(rows, columns, figsize=(2*rows, 2*columns),
                            sharey=True)
    for class_idx in class_range:
        for feature_idx in range(num_features):
            weights_normalized = w[feature_idx][0:-1][:, class_idx] - w[feature_idx][0:-1].mean(axis=-1)
            if num_classes > 2:
                ax = axs[class_idx][feature_idx]
            else:
                ax = axs.ravel()[feature_idx]
            bin_edge = np.concatenate(
                [[bin_edges[feature_idx][0] - (bin_edges[feature_idx][1] - bin_edges[feature_idx][0])],
                 bin_edges[feature_idx]])
            # Compute the midpoints of the bin edges
            ax.step(
                bin_edge, w[feature_idx][:, class_idx] - w[feature_idx][:, class_idx].mean(), where='pre')>>>>>>> develop
            if class_idx == 0:
                if feature_names is None:
                    ax.set_title(f'Feature {feature_idx}')
                else:
                    ax.set_title(feature_names[feature_idx])

            if feature_idx == 0:
                ax.set_ylabel(f'Class {class_idx}')
    if num_classes == 2:
        for i in range(num_features, len(axs.ravel())):
            axs.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
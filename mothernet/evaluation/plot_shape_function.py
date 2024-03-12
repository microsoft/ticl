import matplotlib.pyplot as plt
import numpy as np


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray):
    num_features = len(bin_edges)
    fig, axs = plt.subplots(1, num_features, figsize=(2*num_features, 2))
    axs = axs.flatten()
    for i, (bin_edge, w) in enumerate(zip(bin_edges, w)):
        axs[0].plot(bin_edge, w.T[0][1:], label=f'Feature {i}')
        axs[1].plot(bin_edge, w.T[1][1:], label=f'Feature {i}')
    axs[0].set_title('Class 0')
    axs[1].set_title('Class 1')
    legend = axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend.set_zorder(102)
    plt.tight_layout()
    plt.show()
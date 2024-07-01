import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams["figure.constrained_layout.use"] = True


def plot_individual_shape_function(models, data_density, dataset_name, feature_names=None, X_train=None):
    colors = {'GAMformer': 'red', 'EBM': 'blue'}
    for feature_idx, feature_name in enumerate(feature_names):
        print(f'Plotting shape function for feature {feature_name}')
        fig, axs_baam = plt.subplots(1, 1, figsize=(2. * 2, 1.85 * 1))
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_density = inset_axes(axs_baam, width="100%", height="25%", loc='lower center',
                                bbox_to_anchor=(0.0, .98, 1.0, 0.25),
                                bbox_transform=axs_baam.transAxes)
        if X_train[feature_name].dtype == 'O' or len(X_train[feature_name].unique()) < 64:
            # categorical
            bars, counts = np.unique(X_train[feature_name], return_counts=True)
            bin_edges = (bars[:-1] + bars[1:]) / 2
            # add left and right
            custom_bin_edges = np.concatenate([np.array([bars[0] - 0.5]), bin_edges, np.array([bars[-1] + 0.5])])
            for i in range(len(counts)):
                ax_density.hlines(counts[i], custom_bin_edges[i], custom_bin_edges[i + 1],
                                  color='black', lw=1.5)
        else:
            if X_train is None:
                ax_density.plot(data_density[feature_name]['names'][:-1], data_density[feature_name]['scores'])
            else:
                kde = gaussian_kde(X_train[feature_name])
                xs = np.linspace(X_train[feature_name].min(), X_train[feature_name].max(), 1000)
                ax_density.plot(xs, kde(xs), color='black', lw=1)
        ax_density.set_xticks([])
        ax_density.set_yticks([])
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        ax_density.spines['bottom'].set_visible(False)
        ax_density.spines['left'].set_visible(False)

        # plot ebm on twin y axis
        axs_ebm = axs_baam.twinx()
        for model_idx, (model_name, model) in enumerate(models.items()):
            for bin_edges, w in zip([split[feature_name] for split in model['bin_edges']],
                                    [split[feature_name] for split in model['w']]):
                if model_name.upper() == 'EBM':
                    ax = axs_ebm
                elif model_name == 'GAMformer':
                    ax = axs_baam
                else:
                    raise NotImplementedError('Only EBM and GAMformer are supported')
                w = np.array(w)
                if model_name == 'GAMformer':
                    w = w[:, 1]
                bin_edges = np.array(bin_edges)
                if X_train[feature_name].dtype == 'O' or len(X_train[feature_name].unique()) < 64:
                    # categorical
                    if model_name.upper() == 'EBM':
                        if len(bin_edges) > 2:
                            bins = bin_edges[1:-1]
                            bins = np.concatenate([[bins[0] - 1], bins, [bins[-1] + 1]])
                            w = w - w.mean()
                            for i in range(len(w)):
                                ax.hlines(w[i], bins[i], bins[i + 1], color=colors[model_name],
                                          alpha=1 / len(model['bin_edges']), lw=3, label=model_name)

                    else:
                        bins, index = np.unique(bin_edges, return_index=True)
                        relevant_weights = w[index]
                        relevant_weights = relevant_weights - relevant_weights.mean()
                        custom_bin_edges = (bins[:-1] + bins[1:]) / 2
                        # Add left and right
                        custom_bin_edges = np.concatenate(
                            [np.array([bins[0] - 0.5]), custom_bin_edges, np.array([bins[-1] + 0.5])])
                        for i in range(len(relevant_weights)):
                            ax.hlines(relevant_weights[i], custom_bin_edges[i], custom_bin_edges[i + 1],
                                      label=model_name,
                                      color=colors[model_name], alpha=1 / len(model['bin_edges']), lw=3)
                else:
                    weights_normalized = w - w.mean(axis=-1)
                    # continuous
                    if model_name == 'GAMformer':
                        # Add bin edges for GAMformer which extend for the data range of X_train
                        bin_edges = np.concatenate([np.array([X_train[feature_name].min()]),
                                                    bin_edges,
                                                    np.array([X_train[feature_name].max()])])
                    # Add weight again for the last bin edge
                    weights_normalized = np.concatenate([weights_normalized, [weights_normalized[-1]]])
                    ax.step(bin_edges, weights_normalized, label=model_name, c=colors[model_name],
                            alpha=1 / len(model['bin_edges']))

        axs_baam.set_ylabel(f'Log-Odds\n(GAMFormer)')
        axs_ebm.set_ylabel(f'Log-Odds\n(EBM)')
        axs_baam.set_xlabel(feature_name)

        # Create a dictionary of custom legend handles with full opacity
        custom_handles = {}
        for model_name in models.keys():
            custom_handles[model_name] = mlines.Line2D([], [], color=colors[model_name], lw=3)

        # add legend with baam and ebm in correct color
        handles, labels = axs_baam.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # also add ebm which is on the twin axis
        handles, labels = axs_ebm.get_legend_handles_labels()
        by_label.update(dict(zip(labels, handles)))

        # Update by_label to use custom handles
        for label in by_label.keys():
            by_label[label] = custom_handles[label]
        legend = plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.3, 1))
        legend.set_zorder(102)
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_shape_functions_{feature_name}.pdf', bbox_inches='tight')
        plt.close()


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
    fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 2 * rows),
                            sharey=True)
    feature_range = feature_subset if feature_subset is not None else range(num_features)
    for class_idx in class_range:
        for ax_idx, feature_idx in enumerate(feature_range):
            weights_normalized = w[feature_idx][0:-1][:, class_idx] - w[feature_idx][0:-1].mean(axis=-1)
            if num_classes > 2:
                ax = axs[class_idx][ax_idx]
            else:
                if columns == 1 and rows == 1:
                    ax = axs
                else:
                    ax = axs.ravel()[ax_idx]
            ax.step(bin_edges[feature_idx], weights_normalized)
            if class_idx == 0 or num_classes == 2:
                if feature_names is None:
                    ax.set_title(f'Feature {feature_idx}')
                else:
                    ax.set_title(f'{feature_names[feature_idx]}')

            if feature_idx == 0:
                ax.set_ylabel(f'Class {class_idx}')
    if num_classes == 2 and rows * columns > 1:
        for i in range(num_features, len(axs.ravel())):
            axs.ravel()[i].set_axis_off()
    plt.tight_layout()
    return axs

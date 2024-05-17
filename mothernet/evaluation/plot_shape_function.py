import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.constrained_layout.use"] = True


def plot_individual_shape_function(models, feature_names=None):
    colors = {'GAMformer': 'red', 'EBM': 'blue'}
    for feature_idx, feature_name in enumerate(feature_names):
        fig, axs_baam = plt.subplots(1, 1, figsize=(1.9 * 2, 2. * 1))
        # plot ebm on twin y axis
        axs_ebm = axs_baam.twinx()
        for model_idx, (model_name, model) in enumerate(models.items()):
            for bin_edges, w in zip([split[feature_name] for split in model['bin_edges']],
                                    [split[feature_name] for split in model['w']]):
                w = np.array(w)
                if model_name == 'GAMformer':
                    w = w[:, 1]
                bin_edges = np.array(bin_edges)
                weights_normalized = w[0:-1] - w[0:-1].mean(axis=-1)
                if model_name.upper() == 'EBM':
                    ax = axs_ebm
                else:
                    ax = axs_baam
                ax.step(bin_edges, weights_normalized, label=model_name, c=colors[model_name],
                        alpha=1 / len(model['bin_edges']))
        axs_baam.set_ylabel(f'Log-Odds\n(GAMFormer)')
        axs_ebm.set_ylabel(f'Log-Odds\n(EBM)')
        axs_baam.set_xlabel(feature_name)
        # add legend with baam and ebm in correct color
        handles, labels = axs_baam.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # also add ebm which is on the twin axis
        handles, labels = axs_ebm.get_legend_handles_labels()
        by_label.update(dict(zip(labels, handles)))
        legend = plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.3, 1))
        legend.set_zorder(102)
        plt.savefig(f'mimic_2_shape_functions_{feature_name}.pdf', bbox_inches='tight')


def plot_shape_function(models, feature_names=None, feature_subset=None):
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

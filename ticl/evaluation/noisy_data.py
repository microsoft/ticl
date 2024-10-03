from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from imbalanced_data import eval_gamformer_and_ebm

if __name__ == '__main__':
    results = defaultdict(list)
    for label_noise in np.linspace(0.0, 1.0, 20):
        for seed in range(10):
            X, y = make_classification(n_samples=300, n_features=20, n_classes=2,
                                       n_clusters_per_class=1, weights=[1 / 2, 1 / 2], random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            # Generate a boolean mask where True means the label should be flipped
            flip_mask = np.random.choice([True, False], size=y_train.shape, p=[label_noise, 1 - label_noise])

            # Use the mask to flip the labels
            y_train[flip_mask] = 1 - y_train[flip_mask]

            res = eval_gamformer_and_ebm('label noise', X_train, y_train, X_test, y_test)
            results['Label Noise'].extend([label_noise, label_noise])
            results['AUC-ROC'].extend([res[0]['test_node_gam_bagging'], res[1]['test_node_gam_bagging']])
            results['Model'].extend(['EBM', 'GAMFormer'])

    data = pd.DataFrame.from_dict(results)
    data.to_csv('noisy_labels.csv')

    plt.figure(figsize=(3.1, 1.6))
    sns.lineplot(data=data, x='Label Noise', y='AUC-ROC', hue='Model')
    legend = plt.gca().legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend.set_zorder(102)
    plt.xlim(min(data['Label Noise']), max(data['Label Noise']))
    plt.tight_layout()
    plt.savefig('label_noise.pdf')

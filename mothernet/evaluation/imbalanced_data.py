from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split

from benchmark_node_gam_datasets import process_model
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.utils import get_mn_model

plt.style.use(['science', 'no-latex', 'light'])


def eval_gamformer_and_ebm(dataset_name, X, y, X_test, y_test, ct=None, n_splits=3, random_state=1337):
    records = []
    summary_record = {}
    summary_record['dataset_name'] = dataset_name
    # Main effects only EBM
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state, interactions=0)
    record = process_model(ebm_inter, 'ebm-main-effects', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    # No pipeline for BAAM
    model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    model_path = get_mn_model(model_string)
    record = process_model(
        MotherNetAdditiveClassifier(device='cpu', path=model_path), 'baam',
        X, y,
        X_test, y_test,
        n_splits=n_splits, n_jobs=1
    )
    print(record)
    record.update(summary_record)
    records.append(record)
    return records


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    results = defaultdict(list)
    for class_2_ratio in np.linspace(1 / 2, 0.95, 15):
        ratios = np.array([1 - class_2_ratio, class_2_ratio])
        for seed in range(15):
            X, y = make_classification(n_samples=200, n_features=20, n_classes=2,
                                       n_clusters_per_class=1, weights=ratios, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            res = eval_gamformer_and_ebm('imbalanced_data', X_train, y_train, X_test, y_test)
            results['Imbalance Ratio'].extend([class_2_ratio, class_2_ratio])
            results['AUC-ROC'].extend([res[0]['test_node_gam_bagging'], res[1]['test_node_gam_bagging']])
            results['Model'].extend(['EBM', 'GAMFormer'])

    data = pd.DataFrame.from_dict(results)
    data.to_csv('imbalanced_data.csv', header=False)

    plt.figure(figsize=(4, 3))
    sns.lineplot(data=data, x='Imbalance Ratio', y='AUC-ROC', hue='Model')
    plt.savefig('imbalanced_data.pdf')

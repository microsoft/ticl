from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
import torch
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.utils import resample

from benchmark_node_gam_datasets import process_model
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.utils import get_mn_model

plt.style.use(['science', 'no-latex', 'light'])


def eval_gamformer_and_ebm(dataset_name, X, y, X_test, y_test, column_names, ct=None, n_splits=3, random_state=1337,
                           record_shape_functions=False):
    records = []
    summary_record = {}
    summary_record['dataset_name'] = dataset_name
    # Main effects only EBM
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state, interactions=0,
                                              feature_names=column_names)
    record = process_model(ebm_inter, 'ebm-main-effects', X, y, X_test, y_test, n_splits=n_splits,
                           record_shape_functions=record_shape_functions)
    print(record)
    record.update(summary_record)
    records.append(record)

    # No pipeline for BAAM
    is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
    cat_cols = X.columns.values[is_cat]
    num_cols = X.columns.values[~is_cat]

    cat_ohe_step = ('ohe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype='int'))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([('identity', FunctionTransformer())])
    transformers = [
        ('cat', cat_pipe, cat_cols),
        ('num', num_pipe, num_cols)
    ]
    ct = ColumnTransformer(transformers=transformers, sparse_threshold=0)

    # No pipeline for BAAM
    model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    model_path = get_mn_model(model_string)
    '''
    baam = Pipeline([
        ('identity', FunctionTransformer()),
        # n_estimators updated from 10 to 100 due to sci-kit defaults changing in future versions
        ('baam', MotherNetAdditiveClassifier(device='cpu', path=model_path)),
    ])
    '''
    baam = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    record = process_model(
        baam, 'baam',
        X.to_numpy().astype(np.float32), y, X_test.to_numpy().astype(np.float32), y_test,
        n_splits=n_splits, n_jobs=1, record_shape_functions=record_shape_functions,
        column_names=column_names
    )
    print(record)
    record.update(summary_record)
    records.append(record)
    return records


if __name__ == '__main__':
    results = defaultdict(list)
    for class_2_ratio in np.linspace(0.5, 0.95, 20):
        ratios = np.array([1 - class_2_ratio, class_2_ratio])
        for seed in range(15):
            X, y = make_classification(n_samples=300, n_features=20, n_classes=2,
                                       n_clusters_per_class=1, weights=ratios, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            res = eval_gamformer_and_ebm('imbalanced_data', X_train, y_train, X_test, y_test)
            results['Imbalance Ratio'].extend([class_2_ratio, class_2_ratio])
            results['AUC-ROC'].extend([res[0]['test_node_gam_bagging'], res[1]['test_node_gam_bagging']])
            results['Model'].extend(['EBM', 'GAMFormer'])

    data = pd.DataFrame.from_dict(results)
    data.to_csv('imbalanced_data.csv')

    plt.figure(figsize=(3.1, 1.6))
    sns.lineplot(data=data, x='Imbalance Ratio', y='AUC-ROC', hue='Model')
    plt.xlim(min(data['Imbalance Ratio']), max(data['Imbalance Ratio']))
    legend = plt.gca().legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend.set_zorder(102)
    plt.tight_layout()
    plt.savefig('imbalanced_data.pdf', bbox_inches='tight')

import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split

from ticl.datasets import linear_correlated_logistic_regression
from ticl.evaluation.imbalanced_data import eval_gamformer_and_ebm
from ticl.evaluation.node_gam_data import DATASETS
from ticl.evaluation.plot_shape_function import plot_individual_shape_function

plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True
plt.savefig('mimic_2_shape_functions.pdf', dpi=300, bbox_inches='tight')


def scaling_analysis(model_string: str, dataset: str):
    # create time stamp with current time in nice string format use datetime.now() and time.strftime()
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    data = DATASETS[dataset]()

    X_train = data['X_train']
    y_train = data['y_train']
    # Scaling Analysis
    os.makedirs(f'output/{dataset}', exist_ok=True)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_train, y_train,
                                                                  train_size=0.8, random_state=42)
    # Iterate over dataset sizes and sample multiple datasets per size to get an average
    results_dict = defaultdict(list)
    for size in np.linspace(100, len(X_train_full) * 0.95, 15):
        ratio = size / X_train_full.shape[0]
        for i in range(3):
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train_full, y_train_full, train_size=ratio, random_state=42 + i)

            res = eval_gamformer_and_ebm('scaling_analysis', X_train_sub, y_train_sub, X_test, y_test,
                                         column_names=data['X_train'].columns, n_splits=1)
            results_dict['size'].extend([size, size])
            results_dict['AUC-ROC'].extend([res[0]['test_node_gam_bagging'], res[1]['test_node_gam_bagging']])
            results_dict['Model'].extend(['EBM', 'GAMFormer'])
            json.dump(results_dict, open(f"output/{dataset}/scaling_analysis_results_{time_stamp}.json", "w"))

            print(
                f"Dataset: {dataset}, Size (ratio): {size}, Size (abs): {X_train_sub.shape[0]}  AUC-ROC (EBM / GAM): {res[0]['test_node_gam_bagging']}, {res[1]['test_node_gam_bagging']}")

    return time_stamp


def plot_shape_functions(model_string: str, dataset: str):
    data = DATASETS[dataset]()

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, train_size=0.95)
    results = eval_gamformer_and_ebm(dataset, X_train, y_train, X_test, y_test, n_splits=30,
                                     column_names=data['X_train'].columns, record_shape_functions=True)
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.makedirs(f'output/{dataset}', exist_ok=True)
    pickle.dump(results, open(f"output/{dataset}/shape_function_results_{time_stamp}.pkl", "wb"))
    # Plot shape function per feature
    feature_columns_non_constant = []
    for feature_name in X_train.columns:
        if len(X_train[feature_name].unique()) > 1:
            feature_columns_non_constant.append(feature_name)
    plot_individual_shape_function(models={'EBM': {'bin_edges': results[0]['bin_edges'], 'w': results[0]['w']},
                                           'GAMformer': {'bin_edges': results[1]['bin_edges'], 'w': results[1]['w']}},
                                   data_density=results[0]['data_density'][0],
                                   feature_names=feature_columns_non_constant, X_train=X_train, dataset_name=dataset)


def toy_datasets():
    # logistic regression
    column_names = [r'$x_1$', r'$x_2$', r'$x_3$']
    X, y = linear_correlated_logistic_regression(
        n_features=3, n_tasks=1, n_datapoints=2000, sampling_correlation=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    results = eval_gamformer_and_ebm('logistic regression', X_train, y_train, X_test, y_test, n_splits=5,
                                     column_names=column_names, record_shape_functions=True)
    plot_individual_shape_function(models={'EBM': {'bin_edges': results[0]['bin_edges'], 'w': results[0]['w']},
                                           'GAMformer': {'bin_edges': results[1]['bin_edges'], 'w': results[1]['w']}},
                                   data_density=results[0]['data_density'][0],
                                   feature_names=column_names, X_train=pd.DataFrame(X_train, columns=column_names),
                                   dataset_name='logistic_regression')

    # polynomial regression
    column_names = [r'$x_1$', r'$x_2$']
    X, _ = linear_correlated_logistic_regression(
        n_features=2, n_tasks=1, n_datapoints=2000, sampling_correlation=0.0)
    y = np.array(sigmoid(X[:, 0] + X[:, 1] ** 2) > 0.5, dtype=np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    results = eval_gamformer_and_ebm('polynomial regression', X_train, y_train, X_test, y_test, n_splits=5,
                                     column_names=column_names, record_shape_functions=True)
    plot_individual_shape_function(models={'EBM': {'bin_edges': results[0]['bin_edges'], 'w': results[0]['w']},
                                           'GAMformer': {'bin_edges': results[1]['bin_edges'], 'w': results[1]['w']}},
                                   data_density=results[0]['data_density'][0],
                                   feature_names=column_names, X_train=pd.DataFrame(X_train, columns=column_names),
                                   dataset_name='polynomial_regression')


if __name__ == '__main__':
    model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    # Run Toy Datasets
    toy_datasets()
    # Shape Function Visualization
    for dataset in ['MIMIC2', 'MIMIC3', 'ADULT', 'SUPPORT2']:
        plot_shape_functions(model_string, dataset)

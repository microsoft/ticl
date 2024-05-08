# Adapted from https://nbviewer.org/github/interpretml/interpret/blob/develop/docs/benchmarks/ebm-classification-comparison.ipynb
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

from mothernet.evaluation.node_gam_data import DATASETS
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.utils import get_mn_model


def load_node_gam_data(dataset_name: str):
    data = DATASETS[dataset_name.upper()]()
    dataset = {
        'problem': data['problem'],
        'full': {
            'X': data["X_train"],
            'y': data["y_train"],
        },
        'test': {
            'X': data["X_test"],
            'y': data["y_test"],
        },
    }

    return dataset


def format_n(x):
    return "{0:.3f}".format(x)


def process_model(clf, name, X, y, X_test, y_test, n_splits=3, test_size=0.25, n_jobs=None):
    # Evaluate model
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=1337)
    print('Fitting', name)
    start = time.time()
    scores = cross_validate(
        clf, X, y, scoring='roc_auc', cv=ss,
        n_jobs=n_jobs, return_estimator=True
    )
    end = time.time()
    print(f'Done: {end - start}, Testing', name)
    n_train_points = X.shape[0] * (1 - test_size)
    n_test_points = X.shape[0] * test_size
    record = dict()
    start = time.time()
    record['model_name'] = name
    record['n_train_points'] = n_train_points
    record['n_test_points'] = n_test_points
    record['fit_time_mean'] = format_n(np.mean(scores['fit_time']))
    record['fit_time_std'] = format_n(np.std(scores['fit_time']))
    record['test_score_mean'] = format_n(np.mean(scores['test_score']))
    record['test_score_std'] = format_n(np.std(scores['test_score']))
    record['test_node_gam_scores'] = [
        roc_auc_score(
            y_test.flatten(),
            estimator.predict_proba(X_test)[:, 1].flatten()) for estimator in scores['estimator']
    ]
    end = time.time()
    print(f'Done: {end - start}')
    return record


def benchmark_models(dataset_name, X, y, X_test, y_test, ct=None, n_splits=3, random_state=1337):
    if ct is None:
        is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        cat_ohe_step = ('ohe', OneHotEncoder(handle_unknown='ignore'))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers, sparse_threshold=0)

    records = []

    summary_record = {}
    summary_record['dataset_name'] = dataset_name
    print()
    print('-' * 78)
    print(dataset_name)
    print('-' * 78)
    print(summary_record)
    print()

    pipe = Pipeline([
        ('ct', ct),
        ('std', StandardScaler()),
        ('lr', LogisticRegression(random_state=random_state)),
    ])
    record = process_model(pipe, 'lr', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline([
        ('ct', ct),
        # n_estimators updated from 10 to 100 due to sci-kit defaults changing in future versions
        ('rf-100', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)),
    ])
    record = process_model(pipe, 'rf-100', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGBClassifier(random_state=random_state, eval_metric='logloss')),
    ])
    record = process_model(pipe, 'xgb', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    # No pipeline needed due to EBM handling string datatypes
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state)
    record = process_model(ebm_inter, 'ebm', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    # Main effects only EBM
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state, interactions=0)
    record = process_model(ebm_inter, 'ebm-main-effects', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    if ct is None:
        is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        cat_ohe_step = ('ohe', OrdinalEncoder(handle_unknown='ignore'))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers, sparse_threshold=0)
    # No pipeline for BAAM
    model_string = "baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_400.cpkt"
    model_path = get_mn_model(model_string)
    baam = Pipeline([
        ('ct', ct),
        # n_estimators updated from 10 to 100 due to sci-kit defaults changing in future versions
        ('baam', MotherNetAdditiveClassifier(device='cpu', path=model_path)),
    ])
    record = process_model(
        baam, 'baam',
        X, y,
        X_test, y_test,
        n_splits=n_splits, n_jobs=1, test_size=0.8
    )
    print(record)
    record.update(summary_record)
    records.append(record)

    return records


results = []
n_splits = 5

time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

for dataset_name in ['Churn', 'Support2', 'Adult', 'mimic2', 'mimic3', 'income',
                     'credit', 'microsoft', 'year']:
    try:
        dataset = load_node_gam_data(dataset_name)
        result = benchmark_models(
            dataset_name,
            dataset['full']['X'], dataset['full']['y'],
            dataset['test']['X'], dataset['test']['y'],
            n_splits=n_splits
        )
        os.makedirs(f"output/{dataset_name}", exist_ok=True)
        json.dump(result, open(f"output/{dataset_name}/node_gam_benchmark_results_{time_stamp}.json", "w"))
        results.append(result)

        records = [item for result in results for item in result]
        record_df = pd.DataFrame.from_records(records)
        record_df.to_csv(f'node_gam_benchmark_results_{time_stamp}.csv')
    except Exception as e:
        print(e)

'''
df = pd.read_csv('ebm-perf-classification-overnight.csv')
for dataset_name, df_dataset in df.groupby('dataset_name'):
    print(f'\nDataset: {dataset_name}')
    for method, df_method in df_dataset.groupby('model_name'):
        l = eval(df_method['test_node_gam_scores'].to_list()[0])
        print(f'{method}: {np.mean(l):.5f} +- {np.std(l) / np.sqrt(len(l)):.5f}')
'''
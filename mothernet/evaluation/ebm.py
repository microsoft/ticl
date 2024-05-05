import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from mothernet.evaluation.node_gam_data import DATASETS


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


def process_model(clf, name, X, y, X_test, y_test, n_splits=3):
    # Evaluate model
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1337)
    scores = cross_validate(
        clf, X, y, scoring='roc_auc', cv=ss,
        n_jobs=None, return_estimator=True
    )

    record = dict()
    record['model_name'] = name
    record['fit_time_mean'] = format_n(np.mean(scores['fit_time']))
    record['fit_time_std'] = format_n(np.std(scores['fit_time']))
    record['test_score_mean'] = format_n(np.mean(scores['test_score']))
    record['test_score_std'] = format_n(np.std(scores['test_score']))
    record['test_split_scores'] = [estimator.score(X_test, y_test) for estimator in scores['estimator']]

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
        ct = ColumnTransformer(transformers=transformers)

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
    record = process_model(pipe, 'lr', X, y,  X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline([
        ('ct', ct),
        # n_estimators updated from 10 to 100 due to sci-kit defaults changing in future versions
        ('rf-100', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)),
    ])
    record = process_model(pipe, 'rf-100', X, y,  X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGBClassifier(random_state=random_state, eval_metric='logloss')),
    ])
    record = process_model(pipe, 'xgb', X, y,  X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    # No pipeline needed due to EBM handling string datatypes
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state)
    record = process_model(ebm_inter, 'ebm', X, y, X_test, y_test, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    return records


results = []
n_splits = 3

for dataset_name in ['mimic2', 'mimic3', 'compas']:
    dataset = load_node_gam_data(dataset_name)
    result = benchmark_models(
        dataset_name,
        dataset['full']['X'], dataset['full']['y'],
        dataset['test']['X'], dataset['test']['y'],
        # ct=dataset['ct'],
        n_splits=n_splits
    )
    results.append(result)

records = [item for result in results for item in result]
record_df = pd.DataFrame.from_records(records)[['dataset_name', 'model_name', 'test_score_mean', 'test_score_std', 'test_split_scores']]
record_df.to_csv('ebm-perf-classification-overnight.csv')

# In[ ]:

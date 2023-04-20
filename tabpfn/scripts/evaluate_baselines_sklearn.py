import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tabpfn.scripts.distill_mlp import TorchMLP, DistilledTabPFNMLP

def make_logreg(categorical_features):
    cont_pipe = make_pipeline(StandardScaler(), SimpleImputer())
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categorical_features), remainder=cont_pipe)
    return make_pipeline(preprocess, LogisticRegression(max_iter=1000))

def make_knn(categorical_features):
    cont_pipe = make_pipeline(StandardScaler(), SimpleImputer())
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features), remainder=cont_pipe)
    return make_pipeline(preprocess, KNeighborsClassifier())

def make_hgb(categorical_features):
    preprocess = make_column_transformer((OrdinalEncoder(), categorical_features), remainder="passthrough")
    return make_pipeline(preprocess, HistGradientBoostingClassifier(categorical_features=categorical_features))

def make_rf(categorical_features):
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features), remainder=SimpleImputer())
    return make_pipeline(preprocess, RandomForestClassifier())

def make_tabpfn(categorical_features):
    cont_pipe = make_pipeline(StandardScaler(), SimpleImputer())
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10), categorical_features), remainder=cont_pipe)
    return make_pipeline(preprocess, TabPFNClassifier())

def make_mlp(categorical_features):
    cont_pipe = make_pipeline(StandardScaler(), SimpleImputer())
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features), remainder=cont_pipe)
    return make_pipeline(preprocess, TorchMLP(n_epochs=100))

def make_distilled_tabpfn(categorical_features):
    cont_pipe = make_pipeline(StandardScaler(), SimpleImputer())
    preprocess = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10), categorical_features), remainder=cont_pipe)
    return make_pipeline(preprocess, DistilledTabPFNMLP(n_epochs=100))

def make_lgbm(categorical_features):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(categorical_features=categorical_features)

def evaluate(previous_results=None, models=None, verbose=0):
    from tqdm import tqdm
    from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification

    cc_valid_datasets_multiclass, cc_valid_datasets_multiclass_df = load_openml_list(open_cc_valid_dids, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)
    if models is None:
        models = {'mlp': make_mlp,
                'distilled_tabpfn': make_distilled_tabpfn,
                'logreg': make_logreg,
                    'knn': make_knn,
                    'hgb': make_hgb,
                    'rf': make_rf,
                    'tabpfn': make_tabpfn}

    if previous_results is None:
        from collections import defaultdict
        all_scores = defaultdict(dict)
    else:
        for model in models:
            if model not in previous_results.columns:
                previous_results[model] = np.NaN
    for ds_name, X, y, categorical_features, _, _ in tqdm(cc_valid_datasets_multiclass):
        if verbose > 0:
            print(ds_name)
        for model_name, model_creator in models.items():
            if verbose > 1:
                print(model_name)
            if previous_results is not None and not np.isnan(previous_results.loc[ds_name, model_name]):
                continue

            clf = model_creator(categorical_features)
            if X.shape[1] > 100:
                X = X[:, :100]
            try:
                scores = cross_validate(clf, X, y, scoring="roc_auc_ovo", error_score="raise")
                score = scores['test_score'].mean()
            except (ValueError, RuntimeError) as e:
                print("Error: ", str(e))
                score = np.NaN
            if previous_results is None:
                all_scores[ds_name][model_name] = score
            else:
                previous_results.loc[ds_name, model_name] = score
    if previous_results is None:
        return pd.DataFrame(all_scores).T
    return previous_results
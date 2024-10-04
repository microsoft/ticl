import matplotlib.pyplot as plt

from ticl.evaluation.baselines import tabular_baselines

import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter("ignore", FutureWarning)  # openml deprecation of array return type
from ticl.datasets import load_openml_list, open_cc_large_dids, open_cc_valid_dids, new_valid_dids
from ticl.evaluation.baselines.tabular_baselines import knn_metric, catboost_metric, logistic_metric, xgb_metric, random_forest_metric, mlp_metric, hyperfast_metric, hyperfast_metric_tuning, resnet_metric, mothernet_init_metric
from ticl.evaluation.tabular_evaluation import evaluate, eval_on_datasets, transformer_metric
from ticl.evaluation import tabular_metrics
from ticl.prediction.tabpfn import TabPFNClassifier
import os
from ticl.evaluation.baselines.distill_mlp import DistilledTabPFNMLP
from ticl.prediction.mothernet import MotherNetClassifier
from functools import partial
from ticl.evaluation.tabular_evaluation import eval_on_datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from ticl.prediction.mothernet import ShiftClassifier, EnsembleMeta, MotherNetClassifier
from sklearn.impute import SimpleImputer
from ticl.prediction.mothernet_additive import MotherNetAdditiveClassifier

from interpret.glassbox import ExplainableBoostingClassifier


from hyperfast import HyperFastClassifier

# transformers don't have max times
import pandas as pd

import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples10000_08_01_2024_20_58_55') # _on_exit.cpkt
parser.add_argument('--fetch_only', action='store_true', default = False)
parser.add_argument('--split_numbers', type = int, default = 1)
parser.add_argument('--dataset_num', type = int, default = 10000)
parser.add_argument('--datasets', type=str, default = 'large', choices = ['large', 'default', 'new'])
parser.add_argument('--epoch', type = str, default = 'on_exit')
parser.add_argument('--overwrite', action='store_true', default = False)
parser.add_argument('--max_features', type = int, default = 5000)
parser.add_argument('--n_samples', type = int, default = 1000000)
parser.add_argument('--n_jobs', type = int, default = 1)
args = parser.parse_args()

if args.datasets == 'large':
    datasets = open_cc_large_dids
elif args.datasets == 'default':
    datasets = open_cc_valid_dids
elif args.datasets == 'new':
    datasets = new_valid_dids
else:
    raise ValueError('Invalid dataset type')

cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(
    datasets[:args.dataset_num], 
    multiclass=True,
    shuffled=True, 
    filter_for_nan=False, 
    max_samples = args.n_samples, 
    num_feats=args.max_features, 
    max_num_classes=100,
    return_capped=True
)

eval_positions = [5000000]
max_features = args.max_features
base_path = os.path.join('./')
# max_times only affect non-nn models, nn models are not affected by max_times
# for non-nn models, when the runtime is longer than the max_time, it should stop
max_times = [10000000]
metric_used = tabular_metrics.auc_metric
task_type = 'multiclass'

device_dict = {
    'resnet': 'cuda',
    'tabpfn': 'cuda',
    'mlp': 'cuda',
    'logistic': 'cuda',
    'ssm_tabpfn': 'cuda',
}

clf_dict= {
    'knn': knn_metric,
    'rf': random_forest_metric,
    'xgb': xgb_metric,
    'logistic': logistic_metric,
    'mlp': mlp_metric,
    'resnet': resnet_metric,
}

if 'ssm_tabpfn' in args.model:
    model_name = 'ssm_tabpfn'
    ssm_tabpfn = TabPFNClassifier(
        device = device_dict.get(model_name, 'cpu'),
        model_string = args.model,
        epoch = str(args.epoch),
        N_ensemble_configurations=3,
    )
    clf_dict[model_name] = ssm_tabpfn
elif 'tabpfn' in args.model:
    model_name = 'tabpfn'
    tabpfn = TabPFNClassifier(
        device = device_dict.get(model_name, 'cpu'),
        model_string = args.model,
        epoch = str(args.epoch),
        N_ensemble_configurations=3,
    )
    clf_dict[model_name] = tabpfn
else:
    model_name = args.model

    

results_baselines = [
    eval_on_datasets(
        'multiclass', 
        clf_dict[model_name], 
        args.model, 
        cc_test_datasets_multiclass, 
        eval_positions=eval_positions, 
        max_times=max_times,
        metric_used=metric_used, 
        split_numbers=list(range(1,args.split_numbers+1)),
        n_samples=args.n_samples, 
        fetch_only=args.fetch_only,
        device=device_dict.get(model_name, 'cpu'),
        base_path=base_path,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
        max_features=max_features,
        pca = False,
    )
]

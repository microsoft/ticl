import matplotlib.pyplot as plt

from mothernet.evaluation.baselines import tabular_baselines

import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter("ignore", FutureWarning)  # openml deprecation of array return type
from mothernet.datasets import load_openml_list, open_cc_large_dids, open_cc_valid_dids, new_valid_dids
from mothernet.evaluation.tabular_evaluation import eval_on_datasets
from mothernet.evaluation import tabular_metrics
from mothernet.prediction.tabpfn import TabPFNClassifier
import os

# transformers don't have max times
import pandas as pd

import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'tabflex') 
parser.add_argument('--fetch_only', action='store_true', default = False)
parser.add_argument('--split_numbers', type = int, default = 5)
parser.add_argument('--dataset_num', type = int, default = 10000)
parser.add_argument('--datasets', type=str, default = 'new', choices = ['large', 'default', 'new'])
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
    'tabfast': 'cuda',
    'tabsmall': 'cuda',
    'tabflex': 'cuda',
}

clf_dict = {}
if args.model in ['tabflex', 'tabsmall', 'tabfast']:
    if args.model == 'tabflex':
        model_string = 'ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40'
        epoch = '1410'
    elif args.model == 'tabfast':
        model_string = 'ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50'
        epoch = '110'
    elif args.model == 'tabsmall':
        model_string = 'ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44'
        epoch = '1210'
    clf_dict[args.model] = TabPFNClassifier(
        device = device_dict.get(args.model, 'cpu'),
        model_string = model_string,
        epoch = epoch,
        N_ensemble_configurations=3,
        dimension_reduction = 'random_proj',
    )
elif 'tabpfn' == args.model:
    model_string = 'prior_diff_real_checkpoint_n_0'
    epoch = '100'
    tabpfn = TabPFNClassifier(
        device = device_dict.get(args.model, 'cpu'),
        model_string = model_string,
        epoch = epoch,
        N_ensemble_configurations=3,
    )
    clf_dict[args.model] = tabpfn

    

results_baselines = [
    eval_on_datasets(
        'multiclass', 
        clf_dict[args.model], 
        args.model, 
        cc_test_datasets_multiclass, 
        eval_positions=eval_positions, 
        max_times=max_times,
        metric_used=metric_used, 
        split_numbers=list(range(1,args.split_numbers+1)),
        n_samples=args.n_samples, 
        fetch_only=args.fetch_only,
        device=device_dict.get(args.model, 'cpu'),
        base_path=base_path,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
        max_features=max_features,
        pca = False,
    )
]

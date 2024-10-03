import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import datetime
import warnings
#warnings.simplefilter("ignore", FutureWarning)  # openml deprecation of array return type
warnings.simplefilter("ignore", UserWarning)  # scikit-learn select k best
warnings.simplefilter("ignore", RuntimeWarning)  # scikit-learn select k best

from mothernet.datasets import load_openml_list, open_cc_valid_dids, open_cc_dids
from mothernet.evaluation.baselines.tabular_baselines import knn_metric, catboost_metric, logistic_metric, xgb_metric, random_forest_metric, mlp_metric, hyperfast_metric, resnet_metric, mothernet_init_metric
from mothernet.evaluation.tabular_evaluation import evaluate, eval_on_datasets, transformer_metric
from mothernet.evaluation import tabular_metrics
from mothernet.prediction.tabpfn import TabPFNClassifier
from mothernet.evaluation.baselines import tabular_baselines

from mothernet.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification

cc_valid_datasets_multiclass, cc_valid_datasets_multiclass_df = load_openml_list(open_cc_valid_dids, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True, classification=True)

import os
eval_positions = [1000]
max_features = 100
n_samples = 2000
base_path = os.path.join('.')
overwrite = False
metric_used = tabular_metrics.auc_metric
task_type = 'multiclass'

from sklearn import set_config
set_config(skip_parameter_validation=True, assume_finite=True)

from mothernet.evaluation.tabular_evaluation import eval_on_datasets
from mothernet.prediction.mothernet import ShiftClassifier, EnsembleMeta, MotherNetClassifier, MotherNetInitMLPClassifier
from mothernet.prediction.mothernet_additive import MotherNetAdditiveClassifier
from mothernet.evaluation.baselines.distill_mlp import DistilledTabPFNMLP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier
from functools import partial
from hyperfast import HyperFastClassifier

import warnings
max_times = [1]
device = "cuda:3"
#device = "cpu"

from sklearn import set_config
set_config(skip_parameter_validation=True, assume_finite=True)

device = "cuda:3"


mn_Dclass_average_03_25_2024_17_14_32_epoch_2910_ohe_ensemble_8_quantile = EnsembleMeta(MotherNetClassifier(path="models_diff/mn_Dclass_average_03_25_2024_17_14_32_epoch_2910.cpkt", device=device, inference_device="cuda:3"), n_estimators=8, onehot=True, power="quantile")


clf_dict= {

    'mn_Dclass_average_03_25_2024_17_14_32_epoch_2910_ohe_ensemble_8_quantile_profile_15': mn_Dclass_average_03_25_2024_17_14_32_epoch_2910_ohe_ensemble_8_quantile,

    }
results_transformers_profiling = [
    eval_on_datasets('multiclass', model, model_name, cc_valid_datasets_multiclass, eval_positions=eval_positions, max_times=max_times,
                     #metric_used=metric_used, split_numbers=[1, 2, 3, 4, 5],
                     metric_used=metric_used, split_numbers=[1],
                     n_samples=n_samples, base_path=base_path, overwrite=False, n_jobs=1, device=device)
    for model_name, model in clf_dict.items()
]
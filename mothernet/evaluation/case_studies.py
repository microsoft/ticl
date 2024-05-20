import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from mothernet.evaluation.imbalanced_data import eval_gamformer_and_ebm
from mothernet.evaluation.node_gam_data import DATASETS
from mothernet.evaluation.plot_shape_function import plot_individual_shape_function
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.utils import get_mn_model
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True
plt.savefig('mimic_2_shape_functions.pdf', dpi=300, bbox_inches='tight')


def fit_model(model_string: str, X_train: np.ndarray, y_train: np.ndarray):
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    return classifier


def scaling_analysis_train_test_points(model_string: str):
    num_train_points = 1000
    X_train = np.random.randn(num_train_points, 20)
    y_train = np.random.randint(0, 2, num_train_points)

    results = defaultdict(list)
    for num_test_points in np.linspace(100, 3000, 5):
        timings = []
        for _ in range(3):
            X_test = np.random.randn(int(num_test_points), 20)
            y_test = np.random.randint(0, 2, int(num_test_points))
            model = fit_model(model_string, X_train, y_train)
            start = time.time()
            model.predict_proba(X_test)[:, 1]
            end = time.time()
            timings.append(end - start)
            results['test_points'].append(num_test_points)
            results['time'].append(end - start)
        print(f"Dataset: {dataset}, Size (ratio): {num_test_points}, Time: {np.mean(timings)}")

    # Plot the results via seaborn
    df = pd.DataFrame.from_dict(results)
    plt.figure()
    sns.lineplot(df, x='test_points', y='time', ci=95)
    plt.title(f'Train-Test Points Scaling Analysis')
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()


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


def plot_scaling_analysis(dataset, time_stamp):
    import matplotlib.pyplot as plt
    results = json.load(open(f"../output/{dataset}/scaling_analysis_results_{time_stamp}.json"))
    results = {'ROC-AUC': results['roc_auc'], '#in-context examples': results['size']}
    df = pd.DataFrame.from_dict(results)

    # use seaborn to plot with ci=95
    plt.figure()
    sns.lineplot(df, x='#in-context examples', y='ROC-AUC', ci=95)
    plt.title(f'{dataset} Scaling Analysis')
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.savefig('scaling_analysis.pdf')


def plot_shape_functions(model_string: str, dataset: str):
    data = DATASETS[dataset]()

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    '''
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, train_size=0.95)
    results = eval_gamformer_and_ebm(dataset, X_train, y_train, X_test, y_test, n_splits=3,
                                     column_names=data['X_train'].columns, record_shape_functions=True)
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.makedirs(f'output/{dataset}', exist_ok=True)
    pickle.dump(results, open(f"output/{dataset}/shape_function_results_{time_stamp}.pkl", "wb"))
    '''
    results = pickle.load(open(f"/Users/siemsj/projects/mothernet/output/MIMIC2/shape_function_results_05_20_2024_00_40_33.pkl", "rb"))
    # Plot shape function per feature
    plot_individual_shape_function(models={'EBM': {'bin_edges': results[0]['bin_edges'], 'w': results[0]['w']},
                                           'GAMformer': {'bin_edges': results[1]['bin_edges'], 'w': results[1]['w']}},
                                   data_density=results[0]['data_density'][0],
                                   feature_names=data['X_train'].columns, X_train=X_train, dataset_name=dataset)



if __name__ == '__main__':
    '''
    mimic_3 = /Users/siemsj/projects/mothernet/output/MIMIC3/scaling_analysis_results_05_18_2024_10_39_40.json
    mimic_2 = /Users/siemsj/projects/mothernet/output/MIMIC2/scaling_analysis_results_05_18_2024_09_36_58.json
    adult = /Users/siemsj/projects/mothernet/output/ADULT/scaling_analysis_results_05_18_2024_12_55_46.json
    support2 = /Users/siemsj/projects/mothernet/output/SUPPORT2/scaling_analysis_results_05_18_2024_13_21_10.json
    df = pd.DataFrame.from_dict(json.load(open('/Users/siemsj/projects/mothernet/output/MIMIC2/scaling_analysis_results_05_18_2024_09_36_58.json', 'r')))
    plt.figure(figsize=(3.2, 1.5))
    sns.lineplot(df, x='size', y='AUC-ROC', hue='Model')
    plt.axvline(500)
    plt.xlim(right=10000)
    legend = plt.gca().legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend.set_zorder(102)
    plt.xlabel('# training data')
    plt.tight_layout()
    plt.savefig('mimic_2_scaling.pdf')
    '''

    model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    dataset = "MIMIC2"
    # scaling_analysis_train_test_points(model_string)
    # Scaling analysis

    # for dataset in ['support2']:
    #    time_stamp = scaling_analysis(model_string, dataset.upper())
    # plot_scaling_analysis(dataset, '05_02_2024_16_08_36')

    # Shape Function Visualization
    for dataset in ['MIMIC2', 'MIMIC3', 'ADULT', 'SUPPORT2']:
        plot_shape_functions(model_string, dataset)

import itertools
import os
import random
import time
from functools import partial

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import nn
from tqdm import tqdm

from sklearn.base import BaseEstimator

from ticl.evaluation import tabular_metrics
from ticl.utils import torch_nanmean
from ticl.prediction.tabpfn import transformer_predict, TabPFNClassifier
from ticl.evaluation.baselines.baseline_prediction_interface import baseline_predict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
import pdb, wandb

def iterative_imputer_pca(X, n_components):
    imputer = IterativeImputer()
    X_imputed = imputer.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_imputed)
    return X_pca

def is_classification(metric_used):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__ or metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        return 'classification'
    elif metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        return -1


def transformer_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, device='cpu', N_ensemble_configurations=3, classifier=None, onehot=False, **kwargs):
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    if onehot:
        ohe = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', max_categories=10,
                                sparse_output=False), cat_features)], remainder=SimpleImputer(strategy="constant", fill_value=0))
        ohe.fit(x)
        x, test_x = ohe.transform(x), ohe.transform(test_x)
        if x.shape[1] > 100:
            if not is_classification(metric_used):
                raise ValueError('feature selection is only supported for classification tasks')
            skb = SelectKBest(k=100).fit(x, y)
            x, test_x = skb.transform(x), skb.transform(test_x)
    elif classifier is not None:
        classifier.cat_features = cat_features

    if classifier is None:
        classifier = TabPFNClassifier(device=device, N_ensemble_configurations=N_ensemble_configurations)
    tick = time.time()
    classifier.fit(x, y)
    fit_time = time.time() - tick
    # print('Train data shape', x.shape, ' Test data shape', test_x.shape)
    tick = time.time()
    if is_classification(metric_used):
        pred = classifier.predict_proba(test_x)
    else:
        pred = classifier.predict(test_x)
    inference_time = time.time() - tick
    times = {'fit_time': fit_time, 'inference_time': inference_time}
    metric = metric_used(test_y, pred)

    return metric, pred, times


def evaluate(
    datasets, 
    n_samples, 
    eval_positions, 
    metric_used, 
    model, 
    device='cpu',
    verbose=False, 
    return_tensor=False, 
    pca = False,
    **kwargs
):
    """
    Evaluates a list of datasets for a model function.

    :param datasets: List of datasets
    :param n_samples: maximum sequence length
    :param eval_positions: List of positions where to evaluate models
    :param verbose: If True, is verbose.
    :param metric_used: Which metric is optimized for.
    :param return_tensor: Wheater to return results as a pytorch.tensor or numpy, this is only relevant for transformer.
    :param kwargs:
    :return:
    """
    overall_result = {'metric_used': tabular_metrics.get_scoring_string(metric_used), 'n_samples': n_samples, 'eval_positions': eval_positions}

    aggregated_metric_datasets, num_datasets = torch.tensor(0.0), 0

    # For each dataset
    for [ds_name, X, y, categorical_feats, _, _] in datasets:
        dataset_n_samples = min(len(X), n_samples)
        if verbose:
            print(f'Evaluating {ds_name} with {len(X)} samples')
            
        if wandb.run is not None:
            wandb.log({'dataset': ds_name, 'n_samples': dataset_n_samples})

        aggregated_metric, num = torch.tensor(0.0), 0
        ds_result = {}

        for eval_position in (eval_positions if verbose else eval_positions):
            if eval_position is None or (2 * eval_position > dataset_n_samples):
                eval_position_real = int(dataset_n_samples * 0.5)
            else:
                eval_position_real = eval_position
            eval_position_n_samples = int(eval_position_real * 2.0)
            
            if wandb.run is not None:
                wandb.log({'inference_train_test_sample_number': eval_position_real})
            
            # r should be 
            # None, outputs, eval_ys, best_configs, time_used
            r = evaluate_position(
                X, 
                y, 
                model=model, 
                categorical_feats=categorical_feats,
                n_samples=eval_position_n_samples, 
                ds_name=ds_name, 
                eval_position=eval_position_real, 
                metric_used=metric_used, 
                device=device,
                verbose=verbose - 1,
                pca = pca,
                **kwargs
            )

            if r is None:
                print('Execution failed', ds_name)
                continue

            _, outputs, ys, best_configs, time_used = r

            if torch.is_tensor(outputs):
                outputs = outputs.to(outputs.device)
                ys = ys.to(outputs.device)

            # WARNING: This leaks information on the scaling of the labels
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)):
                ys = (ys - torch.min(ys, axis=0)[0]) / (torch.max(ys, axis=0)[0] - torch.min(ys, axis=0)[0])

            # If we use the bar distribution and the metric_used is r2 -> convert buckets
            #  metric used is prob -> keep
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)) and (
                    metric_used == tabular_metrics.r2_metric or metric_used == tabular_metrics.root_mean_squared_error_metric):
                ds_result[f'{ds_name}_bar_dist_at_{eval_position}'] = outputs
                outputs = model.criterion.mean(outputs)

            ys = ys.T
            ds_result[f'{ds_name}_best_configs_at_{eval_position}'] = best_configs
            ds_result[f'{ds_name}_outputs_at_{eval_position}'] = outputs
            ds_result[f'{ds_name}_ys_at_{eval_position}'] = ys
            ds_result[f'{ds_name}_time_at_{eval_position}'] = time_used
            
            new_metric = torch_nanmean(torch.stack([metric_used(ys[i], outputs[i]) for i in range(ys.shape[0])]))

            if not return_tensor:
                def make_scalar(x): return float(x.detach().cpu().numpy()) if (torch.is_tensor(x) and (len(x.shape) == 0)) else x
                new_metric = make_scalar(new_metric)
                ds_result = {k: make_scalar(ds_result[k]) for k in ds_result.keys()}

            lib = torch if return_tensor else np
            if not lib.isnan(new_metric).any():
                aggregated_metric, num = aggregated_metric + new_metric, num + 1

        overall_result.update(ds_result)
        if num > 0:
            aggregated_metric_datasets, num_datasets = (aggregated_metric_datasets + (aggregated_metric / num)), num_datasets + 1

    overall_result['mean_metric'] = aggregated_metric_datasets / num_datasets

    return overall_result


"""
===============================
INTERNAL HELPER FUNCTIONS
===============================
"""


def _eval_single_dataset_wrapper(**kwargs):
    max_time = kwargs['max_time']
    metric_used = kwargs['metric_used']
    time_string = '_time_'+str(max_time) if max_time else ''
    metric_used_string = '_' + tabular_metrics.get_scoring_string(metric_used, usage='') if kwargs['append_metric'] else ''
    result = evaluate(method=kwargs['model_name']+time_string+metric_used_string, **kwargs)
    result['model'] = kwargs['model_name']
    result['dataset'] = kwargs['datasets'][0][0]
    result['split'] = kwargs['split_number']
    result['max_time'] = kwargs['max_time']
    return result


def eval_on_datasets(
    task_type, 
    model, 
    model_name, 
    datasets, 
    eval_positions, 
    max_times, 
    metric_used, 
    split_numbers, 
    n_samples, 
    base_path, 
    overwrite=False, 
    append_metric=True,
    fetch_only=False, 
    verbose=0, 
    n_jobs=-1,
    device='auto', 
    save=True,
    max_features = 100,
    pca = False,
):
    if callable(model):
        model_callable = model
        if device == 'auto':
            device = 'cpu'
    elif isinstance(model, BaseEstimator):
        model_callable = partial(transformer_metric, classifier=model)
        device_param = [v for k, v in model.get_params().items() if "device" in k]
        if device == "auto":
            device = device_param[0] if len(device_param) > 0 else "cpu"
    else:
        raise ValueError(f"Got model {model} of type {type(model)} which is not callable or a BaseEstimator")
    print(f"evaluating {model_name} on {device}")
    if "cuda" in device:
        results = []
        tqdm_bar = tqdm(list(itertools.product(datasets, max_times, split_numbers)))
        for (ds, max_time, split_number) in tqdm_bar:
            tqdm_bar.set_description(f"evaluating {model_name} on {device} {ds[0]}")
            result = _eval_single_dataset_wrapper(
                datasets=[ds], 
                model=model_callable, 
                model_name=model_name, 
                n_samples=n_samples, 
                base_path=base_path, 
                eval_positions=eval_positions,
                device=device, 
                max_splits=1,
                overwrite=overwrite, 
                save=save,
                metric_used=metric_used, 
                path_interfix=task_type, 
                fetch_only=fetch_only,
                split_number=split_number, 
                verbose=verbose, 
                max_time=max_time, 
                append_metric=append_metric,
                max_features = max_features,
                pca = pca,
            )

            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=2)(delayed(_eval_single_dataset_wrapper)(
            datasets=[ds], model=model_callable, model_name=model_name, n_samples=n_samples, base_path=base_path,
            eval_positions=eval_positions, device=device, max_splits=1, overwrite=overwrite,
            save=save, metric_used=metric_used, path_interfix=task_type, fetch_only=fetch_only, split_number=split_number,
            verbose=verbose, max_time=max_time, append_metric=append_metric, max_features = max_features) for ds in datasets for max_time in max_times for split_number in split_numbers)
    return results


def check_file_exists(path):
    """Checks if a pickle file exists. Returns None if not, else returns the unpickled file."""
    if (os.path.isfile(path)):
        # print(f'loading results from {path}')
        with open(path, 'rb') as f:
            return np.load(f, allow_pickle=True).tolist()
    return None


def generate_valid_split(X, y, n_samples, eval_position, is_classification, split_number=1):
    """Generates a deteministic train-(test/valid) split. Both splits must contain the same classes and all classes in
    the entire datasets. If no such split can be sampled in 7 passes, returns None.

    :param X: torch tensor, feature values
    :param y: torch tensor, class values
    :param n_samples: Number of samples in train + test
    :param eval_position: Number of samples in train, i.e. from which index values are in test
    :param split_number: The split id
    :return:
    """
    done, seed = False, 13
    generator = torch.Generator(device=X.device)
    generator.manual_seed(split_number)
    perm = torch.randperm(X.shape[0], generator=generator) if split_number > 1 else torch.arange(0, X.shape[0])
    X, y = X[perm], y[perm]
    old_random_state = random.getstate()
    while not done:
        if seed > 20:
            return None, None  # No split could be generated in 7 passes, return None
        random.seed(seed)
        i = random.randint(0, len(X) - n_samples) if len(X) - n_samples > 0 else 0
        y_ = y[i:i + n_samples]

        if is_classification:
            # Checks if all classes from dataset are contained and classes in train and test are equal (contain same
            # classes) and
            done = len(torch.unique(y_)) == len(torch.unique(y))
            done = done and torch.all(torch.unique(y_) == torch.unique(y))
            done = done and len(torch.unique(y_[:eval_position])) == len(torch.unique(y_[eval_position:]))
            done = done and torch.all(torch.unique(y_[:eval_position]) == torch.unique(y_[eval_position:]))
            seed = seed + 1
        else:
            done = True

    random.setstate(old_random_state)
    eval_xs = torch.stack([X[i:i + n_samples].clone()], 1)
    eval_ys = torch.stack([y[i:i + n_samples].clone()], 1)

    return eval_xs, eval_ys


def evaluate_position(
    X, 
    y, 
    categorical_feats, 
    model, 
    n_samples, 
    eval_position, 
    overwrite, 
    save, 
    base_path, 
    path_interfix, 
    method, 
    ds_name, 
    fetch_only=False,
    max_time=300, 
    split_number=1,
    metric_used=None, 
    device='cpu', 
    verbose=0, 
    pca = False,
    **kwargs,
):
    """
    Evaluates a dataset with a 'n_samples' number of training samples.

    :param X: Dataset X
    :param y: Dataset labels
    :param categorical_feats: Indices of categorical features.
    :param model: Model function
    :param n_samples: Sequence length.
    :param eval_position: Number of training samples.
    :param overwrite: Wheater to ove
    :param overwrite: If True, results on disk are overwritten.
    :param save:
    :param path_interfix: Used for constructing path to write on disk.
    :param method: Model name.
    :param ds_name: Datset name.
    :param fetch_only: Wheater to calculate or only fetch results.
    :param kwargs:
    :return:
    """
    path = os.path.join(base_path, f'results/tabular/{path_interfix}/results_{method}_{ds_name}_{eval_position}_{n_samples}_{split_number}_{device}.npy')
    # log_path =
    # Load results if on disk
    if not overwrite:
        result = check_file_exists(path)
        if result is not None:
            print(f'Loaded saved result for {path}')
            return result
        elif fetch_only:
            print(f'Could not load saved result for {path}')
            return None
        
    # X: (n_samples, n_features) -> (n_samples, n_components)
    if(X.shape[1] > kwargs['max_features']) and pca:
        print(f"| Reducing features from {X.shape[1]} to {kwargs['max_features']}")
        start = time.time()
        xs_pca = iterative_imputer_pca(X.numpy(), n_components=kwargs['max_features'])
        X = torch.from_numpy(xs_pca).float()
        end = time.time()
        print(f"| Done, time elapsed {end-start:.3f}s.")

    # Generate data splits
    eval_xs, eval_ys = generate_valid_split(
        X, 
        y, 
        n_samples, 
        eval_position, 
        is_classification=tabular_metrics.is_classification(metric_used),
        split_number=split_number
    )
    
    
    if eval_xs is None:
        print(f"No dataset could be generated {ds_name} {n_samples}")
        return None

    eval_ys = (eval_ys > torch.unique(eval_ys).unsqueeze(0)).sum(axis=1).unsqueeze(-1)

    if isinstance(model, nn.Module):
        model = model.to(device)
        eval_xs = eval_xs.to(device)
        eval_ys = eval_ys.to(device)

    start_time = time.time()

    if isinstance(model, nn.Module):  # Two separate predict interfaces for transformer and baselines
        # max_time does not affect nn models
        outputs, best_configs = transformer_predict(
            model, 
            eval_xs, 
            eval_ys, 
            eval_position, 
            metric_used=metric_used, 
            categorical_feats=categorical_feats,
            inference_mode=True, 
            device=device, 
            extend_features=True, 
            verbose=verbose,
            **kwargs
        ), None
    else:
        _, outputs, best_configs = baseline_predict(
            model, 
            eval_xs, 
            eval_ys, 
            categorical_feats, 
            eval_pos=eval_position,
            device=device, 
            max_time=max_time, 
            metric_used=metric_used, 
            verbose=verbose, 
            **kwargs
        )
    eval_ys = eval_ys[eval_position:]
    if outputs is None:
        print('Execution failed', ds_name)
        return None

    if torch.is_tensor(outputs):  # Transfers data to cpu for saving
        outputs = outputs.cpu()
        eval_ys = eval_ys.cpu()

    ds_result = None, outputs, eval_ys, best_configs, time.time() - start_time

    if save:
        with open(path, 'wb') as f:
            np.save(f, np.asarray(ds_result, dtype=object))
            if verbose > 0:
                print(f'saved results to {path}')

    return ds_result

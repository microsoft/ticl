import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances


def make_data(n_classes, n_samples, n_steps):
    classes = (np.random.randint(0, n_classes) + np.cumsum(1 - 2 * np.random.randint(0, 2, size=n_steps))) % n_classes
    steps = np.sort(np.random.uniform(size=n_steps - 1))
    samples = np.random.uniform(size=n_samples)
    return samples.reshape(-1, 1), classes[np.searchsorted(steps, samples)]


def make_binary_data(length=10, subset_size=5, n_prototypes=5):
    strings = [list(np.binary_repr(x, width=length)) for x in np.arange(2**length)]
    bits = np.array(strings).astype(int)
    np.random.shuffle(bits)
    features = np.random.permutation(length)[:subset_size]
    dist = pairwise_distances(bits[:, features], bits[:n_prototypes][:, features]).min(axis=1)
    labels = (dist >= np.median(dist)).ravel()
    if len(np.unique(labels)) == 1:
        labels = dist == 0
    return bits, labels


def get_scores(length, subset_size, n_prototypes, models):
    from sklearn.model_selection import cross_validate
    X, y = make_binary_data(length=length, subset_size=subset_size, n_prototypes=n_prototypes)
    result = {'prototypes': n_prototypes}
    for model_name, model in models.items():
        result[model_name] = np.mean(cross_validate(model, X, y)['test_score'])
    return result


def function_of_rank(rank=2, length=4):
    inputs = np.array(np.meshgrid(*[[-1, 1]]*length)).T.reshape(-1, length)
    outputs = np.zeros(2**length, dtype=bool)

    while 3 * outputs.sum() < len(inputs):
        selected_bits = np.random.choice(length, size=rank, replace=False)
        signs = np.random.choice([-1, 1], size=rank)
        outputs = outputs + ((signs * inputs[:, selected_bits]) == 1).all(axis=1)
    return (inputs + 1) / 2, outputs


def get_scores_rank(length, rank, models):
    from sklearn.model_selection import cross_validate, StratifiedKFold

    X, y = function_of_rank(length=length, rank=rank)
    result = {'rank': rank}
    for model_name, model in models.items():
        result[model_name] = np.mean(cross_validate(model, X, y, cv=StratifiedKFold(shuffle=True))['test_score'])
    return result


def compare_models():
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from ticl.prediction.tabpfn import TabPFNClassifier

    torch.set_num_threads(1)
    prototypes = np.arange(1, 200, 5)
    models = {
        'mlp': MLPClassifier(max_iter=1000),
        'tabpfn': TabPFNClassifier(),
        'rf': RandomForestClassifier()
    }
    res_subset_7 = Parallel(n_jobs=-1)(delayed(get_scores)(length=10, subset_size=7, n_prototypes=n_prototypes, models=models)
                                       for i in range(5) for n_prototypes in prototypes)
    res_subset_7 = pd.DataFrame.from_dict(res_subset_7)
    sns.lineplot(data=res_subset_7.melt(id_vars="prototypes", var_name="model", value_name="score"), x="prototypes", y="score", hue="model")

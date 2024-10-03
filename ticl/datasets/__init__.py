import numpy as np
import openml
import pandas as pd
import torch
from scipy.special import expit as sigmoid


def linear_correlated_logistic_regression(
        n_features: int,
        n_tasks: int,
        n_datapoints: int,
        seed: int | None = 42,
        sampling_correlation: float = 0.0,
        weights: np.array = None,
        *args,
        **kwargs,
) -> tuple[np.array, np.array]:
    """Sample features with linear contribution that are correlated."""
    if weights is None:
        weights = np.array([np.linspace(-1, 1 * i, n_features) for i in range(1, n_tasks + 1)])
    else:
        weights = weights.reshape((n_tasks, n_features))

    if seed is not None:
        np.random.seed(seed)

    inputs_correlated = np.array([np.linspace(-2, 2, n_datapoints) for _ in range(n_features)]).T
    inputs_uniform = np.random.uniform(-2, 2, size=(n_datapoints, n_features))
    inputs = sampling_correlation * inputs_correlated + (1 - sampling_correlation) * inputs_uniform

    condition_number = np.linalg.cond(inputs.T @ inputs)
    targets = np.array(sigmoid(weights.dot(inputs.T)) > 0.5, dtype=np.float64)
    y, X = targets.flatten(), inputs
    return X, y


def linear_correlated_step_function(
        n_features: int,
        n_tasks: int,
        n_datapoints: int,
        seed: int | None = 42,
        sampling_correlation: float = 0.0,
        weights: np.array = None,
        plot: bool = False,
        *args,
        **kwargs,
) -> tuple[np.array, np.array]:
    """Sample features with linear contribution that are correlated."""
    if weights is None:
        weights = np.array([np.linspace(-1, 1 * i, n_features) for i in range(1, n_tasks + 1)])
    else:
        weights = weights.reshape((n_tasks, n_features))

    if seed is not None:
        np.random.seed(seed)

    inputs_correlated = np.array([np.linspace(-2, 2, n_datapoints) for _ in range(n_features)]).T
    inputs_uniform = np.random.uniform(-2, 2, size=(n_datapoints, n_features))
    inputs = sampling_correlation * inputs_correlated + (1 - sampling_correlation) * inputs_uniform

    steps = np.linspace(np.min(inputs_uniform) * 0.4, np.max(inputs_uniform) * 0.4, n_features)
    transformed_inputs = np.copy(inputs)
    for feature_i in range(n_features):
        transformed_inputs[:, feature_i] = (inputs[:, feature_i] > steps[feature_i]).astype(np.int32)

    targets = (transformed_inputs.sum(axis=1) > 0).astype(np.float32)
    condition_number = np.linalg.cond(inputs.T @ inputs)
    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4, 2))
        axs.flatten()
        for i in range(n_features):
            axs[i].scatter(inputs[:, i], transformed_inputs[:, i], s=2)
            axs[i].set_title(f'Class {i}')
        plt.show()
    y, X = targets, inputs
    return X, y


def _encode_if_category(column: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    # copied from old OpenML Python adapter to maintain comparison with tabpfn
    if column.dtype.name == "category":
        column = column.cat.codes.astype(np.float32)
        mask_nan = column == -1
        column[mask_nan] = np.nan
    return column


def get_openml_regression(did, max_samples, shuffled=True):
    dataset = openml.datasets.get_dataset(did, download_data=False, download_qualities=False,
                                          download_features_meta_data=False)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,
                                                                    dataset_format="dataframe")
    try:
        X = np.array(X.apply(_encode_if_category), dtype=np.float32)
    except ValueError as e:
        print(e)
    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(
            0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]
    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def get_openml_classification(did, max_samples, multiclass=True, shuffled=True):
    dataset = openml.datasets.get_dataset(did, download_data=False, download_qualities=False, download_features_meta_data=False)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    X = np.array(X.apply(_encode_if_category), dtype=np.float32)
    y_categorical = y.dtype.name == "category"
    y = np.array(_encode_if_category(y), dtype=int if y_categorical else np.float32)
    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if multiclass and not shuffled:
        raise NotImplementedError("This combination of multiclass and shuffling isn't implemented")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print('Not a NP Array, skipping')
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(
            0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def load_openml_list(
    dids, 
    classification=True, 
    filter_for_nan=False, 
    num_feats=100, 
    min_samples=100, 
    max_samples=400,
    multiclass=True, 
    max_num_classes=10, 
    shuffled=True, 
    return_capped=False, 
    verbose=0,
):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist['NumberOfInstancesWithMissingValues'] == 0]
        print(f'Number of datasets after Nan and feature number filtering: {len(datalist)}')

    for ds in datalist.index:
        modifications = {'samples_capped': False, 'classes_capped': False, 'feats_capped': False}
        entry = datalist.loc[ds]
        if verbose > 0:
            print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, attribute_names = get_openml_classification(
                int(entry.did), max_samples, multiclass=multiclass, shuffled=shuffled)
        if X is None:
            continue

        if X.shape[1] > num_feats:
            if return_capped:
                X = X[:, 0:num_feats]
                categorical_feats = [c for c in categorical_feats if c < num_feats]
                modifications['feats_capped'] = True
            else:
                print('Too many features')
                continue
        if X.shape[0] == max_samples:
            modifications['samples_capped'] = True

        if X.shape[0] < min_samples:
            print(f'Too few samples left for dataset {ds}')
            continue

        if classification:
            if len(np.unique(y)) > max_num_classes:
                if return_capped:
                    X = X[y < np.unique(y)[max_num_classes]]
                    y = y[y < np.unique(y)[max_num_classes]]
                    modifications['classes_capped'] = True
                else:
                    print('Too many classes')
                    continue

        datasets += [[entry['name'], X, y, categorical_feats, attribute_names, modifications]]

    return datasets, datalist


# Classification
valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
test_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163,
                            41147, 1111, 41164, 1169, 1486, 41143, 1461,
                            41167, 40668, 41146, 41169, 41027, 23517,
                            41165, 41161, 41159, 41138, 1590, 41166, 1464,
                            41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
valid_large_classification = [943, 23512, 49, 838, 1131, 767, 1142, 748,
                              1112, 1541, 384, 912, 1503, 796, 20, 30, 903, 4541,
                              961, 805, 1000, 4135, 1442, 816, 1130, 906, 1511,
                              184, 181, 137, 1452, 1481, 949, 449, 50, 913,
                              1071, 831, 843, 9, 896, 1532, 311, 39, 451,
                              463, 382, 778, 474, 737, 1162, 1538, 820, 188,
                              452, 1156, 37, 957, 911, 1508, 1054, 745, 1220,
                              763, 900, 25, 387, 38, 757, 1507, 396, 4153,
                              806, 779, 746, 1037, 871, 717, 1480, 1010, 1016,
                              981, 1547, 1002, 1126, 1459, 846, 837, 1042, 273,
                              1524, 375, 1018, 1531, 1458, 6332, 1546, 1129, 679,
                              389]

open_cc_dids = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469,
                1049, 1050, 1063, 1068, 1510, 1494, 1480, 1462, 1464, 6332,
                23381, 40966, 40982, 40994, 40975]
# Filtered by N_samples < 2000, N feats < 100, N classes < 10
# removed flags 285
open_cc_valid_dids = [
    13, 25, 35, 40, 41, 43, 48, 49, 51, 53, 55, 56, 59, 61, 187, 329, 333, 334, 335, 336, 337, 338, 377, 446,
    450, 451, 452, 460, 463, 464, 466, 470, 475, 481, 679, 694, 717, 721, 724, 733, 738, 745, 747, 748, 750,
    753, 756, 757, 764, 765, 767, 774, 778, 786, 788, 795, 796, 798, 801, 802, 810, 811, 814, 820, 825, 826,
    827, 831, 839, 840, 841, 844, 852, 853, 854, 860, 880, 886, 895, 900, 906, 907, 908, 909, 915, 925, 930,
    931, 934, 939, 940, 941, 949, 966, 968, 984, 987, 996, 1048, 1054, 1071, 1073, 1100, 1115, 1412, 1442,
    1443, 1444, 1446, 1447, 1448, 1451, 1453, 1488, 1490, 1495, 1498, 1499, 1506, 1508, 1511, 1512, 1520,
    1523, 4153, 23499, 40496, 40646, 40663, 40669, 40680, 40682, 40686, 40690, 40693, 40705, 40706, 40710,
    40711, 40981, 41430, 41538, 41919, 41976, 42172, 42261, 42544, 42585, 42638]

# YZ added 
open_cc_large_dids = [
    41168, 
    4534, 
    40668, 
    4135, 
    41027, 
    1461, 
    1590, 
    41162, 
    42733, 
    42734,
    137,
    843,
    846,
    981, 
    1220,
    1459,
    1531,
    1532,
    4135,
    23512,
]

# YZ added
# class ~102, # instances ~850K, # features ~3.1K
new_valid_dids = [311,
 742,
 825,
 841,
 920,
 940,
 1515,
 1549,
 40693,
 40705,
 833,
 1039,
 1491,
 1492,
 1541,
 40645,
 40677,
 41082,
 41144,
 42193,
 279,
 981,
 1536,
 40922,
 40985,
 41986,
 41988,
 41989,
 41990,
 42343,
 1503,
 # 4541, commented because of the overflow error
 40672,
 41991,
 42206]

tabzilla_full_dids = [3,
 6,
 11,
 12,
 14,
 15,
 16,
 18,
 22,
 23,
 28,
 29,
 31,
 32,
 37,
 44,
 46,
 50,
 54,
 151,
 182,
 188,
 307,
 300,
 458,
 469,
 554,
 1049,
 1050,
 1053,
 1063,
 1067,
 1068,
 1590,
 4134,
 1510,
 1489,
 1494,
 1497,
 1501,
 1480,
 1485,
 1486,
 1487,
 1468,
 1475,
 1462,
 1464,
 4534,
 6332,
 1461,
 4538,
 1478,
 23381,
 40499,
 40668,
 40966,
 40982,
 40994,
 40983,
 40975,
 40984,
 40979,
 40996,
 41027,
 23517,
 40923,
 40927,
 40978,
 40670,
 40701,
 44025,
 1596,
 1119,
 4135,
 40685,
 23512,
 40981,
 41169,
 41168,
 41166,
 41165,
 41150,
 41159,
 41161,
 41138,
 41142,
 41163,
 41164,
 41143,
 41146,
 1169,
 41167,
 41147,
 1044,
 5,
 1502,
 42727,
 41145,
 1477,
 1471,
 821,
 344,
 43466,
 23515,
 42,
 13,
 43,
 497,
 171,
 39,
 4535,
 1592,
 1099,
 42183,
 1,
 1509,
 1483,
 41434,
 7,
 9,
 24,
 25,
 27,
 30,
 35,
 40,
 41,
 49,
 51,
 55,
 61,
 1455,
 258,
 43611,
 1568,
 334,
 42855,
 1567,
 42184,
 40733,
 40683,
 163,
 1473,
 477,
 329,
 40664,
 40679,
 59,
 1118,
 1043,
 1479,
 312,
 1116,
 1038,
 1120,
 1036,
 40536,
 451,
 470,
 40496,
 1493,
 377,
 478,
 1459,
 375,
 1466,
 23380,
 1476,
 4,
 40900,
 934,
 782,
 871,
 174,
 184,
 867,
 885,
 736,
 846,
 875,
 916,
 48,
 754,
 448,
 10,
 8,
 194,
 566,
 560,
 189,
 703,
 524,
 562]
 

open_cc_valid_dids_regression = [8, 204, 210, 560, 194, 566, 189, 562, 507, 198, 42821]

grinzstjan_categorical_regression = [44054, 44055, 44056, 44057, 44059, 44061,
                                     44062, 44063, 44064, 44065, 44066, 44068,
                                     44069]

grinzstjan_numerical_classification = [44089, 44090, 44091, 44120, 44121,
                                       44122, 44123, 44124, 44125, 44126,
                                       44127, 44128, 44129, 44130, 44131]

grinzstjan_categorical_classification = [44156, 44157, 44159, 44160, 44161, 44162, 44186]

regression_test_dids = [
    41021,
    416,
    41980,
    422,
    42563,
    42570,
    42726,
    42727,
    42730,
    43071,
    44957,
    44958,
    44959,
    44960,
    44965,
    44967,
    44970,
    44973,
    44978,
    44980,
    44981,
    44994,
    505,
    507,
    531,
    541,
    546,
    550]

"""
===============================
Metrics calculation
===============================
Includes a few metric as well as functions composing metrics on results files.

"""


import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, mean_absolute_error,
                             r2_score, roc_auc_score)


def root_mean_squared_error_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.sqrt(torch.nn.functional.mse_loss(target, pred))


def mean_squared_error_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.nn.functional.mse_loss(target, pred)


def mean_absolute_error_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.tensor(mean_absolute_error(target, pred))


"""
===============================
Metrics calculation
===============================
"""


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(lib.unique(target)) > 2:
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
        return roc_auc_score(target, pred, multi_class=multi_class)
    else:
        if len(pred.shape) == 2:
            pred = pred[:, 1]
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred))
        return roc_auc_score(target, pred)


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


def brier_score_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    target = torch.nn.functional.one_hot(target, num_classes=len(torch.unique(target)))
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    diffs = (pred - target)**2
    return torch.mean(torch.sum(diffs, axis=1))


def ece_metric(target, pred):
    import torchmetrics
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torchmetrics.functional.calibration_error(pred, target)


def average_precision_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(average_precision_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(average_precision_score(target, pred[:, 1] > 0.5))


def balanced_accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(balanced_accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(balanced_accuracy_score(target, pred[:, 1] > 0.5))


def cross_entropy(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        ce = torch.nn.CrossEntropyLoss()
        return ce(pred.float(), target.long())
    else:
        bce = torch.nn.BCELoss()
        return bce(pred[:, 1].float(), target.float())


def r2_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.tensor(neg_r2(target, pred))


def neg_r2(target, pred):
    return -r2_score(pred.float(), target.float())


def is_classification(metric_used):
    if metric_used.__name__ in ["auc_metric", "cross_entropy"]:
        return True
    return False

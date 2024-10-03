
import numpy as np
import pytest

from ticl.prediction import MotherNetClassifier, EnsembleMeta
from ticl.utils import get_mn_model

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch


@pytest.fixture(autouse=True)
def set_threads():
    return torch.set_num_threads(1)


def test_mothernet_paper():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    classifier = MotherNetClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_mothernet_ensemble():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    classifier = EnsembleMeta(MotherNetClassifier(device='cpu', path=model_path), n_estimators=8, power="quantile")
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9

    assert len(classifier.estimators_) == 8


@pytest.mark.parametrize("categorical", [True, False])
def test_mothernet_preprocessing_ensemble(categorical):
    rng = np.random.RandomState(42)
    X = rng.rand(100, 10)
    X[:, 0] = np.NAN
    X[0, 1] = np.NAN
    X[:, 2] = rng.randint(0, 10, 100)
    X[:, 3] = rng.randint(0, 3, 100)
    X[0, 3] = np.NAN
    y = X[:, 8] > 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    cat_features = [2, 3] if categorical else None
    classifier = EnsembleMeta(MotherNetClassifier(device='cpu', path=model_path), n_estimators=32,
                              onehot=True, cat_features=cat_features)
    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)
    assert (prob.argmax(axis=1) == pipeline.predict(X_test)).all()
    assert pipeline.score(X_test, y_test) > 0.9

    assert len(classifier.estimators_) == 32


def test_mothernet_preprocessing_ensemble_all_categorical():
    rng = np.random.RandomState(42)
    X = rng.randint(0, 10, size=(100, 5))
    y = X[:, 3] > 5
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    cat_features = np.arange(5)
    classifier = EnsembleMeta(MotherNetClassifier(device='cpu', path=model_path), n_estimators=32,
                              onehot=True, cat_features=cat_features)
    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)
    assert (prob.argmax(axis=1) == pipeline.predict(X_test)).all()
    assert pipeline.score(X_test, y_test) > 0.9

    assert len(classifier.estimators_) == 32


def test_mothernet_preprocessing_categorical_pruning():
    rng = np.random.RandomState(42)
    X = rng.rand(100, 99)
    X[:, 2] = rng.randint(0, 10, 100)
    X[:, 4] = rng.randint(0, 10, 100)
    y = X[:, 8] > 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    cat_features = [2, 4]
    classifier = EnsembleMeta(MotherNetClassifier(device='cpu', path=model_path), n_estimators=32,
                              onehot=True, cat_features=cat_features)
    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)
    assert (prob.argmax(axis=1) == pipeline.predict(X_test)).all()
    assert pipeline.score(X_test, y_test) > 0.9

    assert len(classifier.estimators_) == 32


if __name__ == "__main__":
    torch.set_num_threads(1)
    from sklearn import set_config
    set_config(skip_parameter_validation=True, assume_finite=True)
    test_mothernet_ensemble()

import numpy as np
import pytest

from mothernet.prediction import MotherNetClassifier, EnsembleMeta
from mothernet.utils import get_mn_model

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    classifier = EnsembleMeta(MotherNetClassifier(device='cpu', path=model_path), n_estimators=3)
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9

    assert len(classifier.vc_.estimators_) == 3


@pytest.mark.parametrize("categorical", [True, False])
def test_mothernet_preprocessing_ensemble(categorical):
    X = np.random.rand(100, 10)
    X[:, 0] = np.NAN
    X[0, 1] = np.NAN
    X[:, 2] = np.random.randint(0, 10, 100)
    X[:, 3] = np.random.randint(0, 3, 100)
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

    assert len(classifier.vc_.estimators_) == 32

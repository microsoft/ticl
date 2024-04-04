
import numpy as np
import pytest

from mothernet.prediction.mothernet import MotherNetInitMLPClassifier
from mothernet.utils import get_mn_model

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def test_mothernet_init_training_no_learning():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    classifier = MotherNetInitMLPClassifier(device='cpu', path=model_path, verbose=10, learning_rate=0.0, n_epochs=1)
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_mothernet_init_training_defaults():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    classifier = MotherNetInitMLPClassifier(device='cpu', path=model_path, verbose=10, n_epochs=100)
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9

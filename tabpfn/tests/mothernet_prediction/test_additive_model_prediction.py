import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabpfn.models.mothernet_additive import ForwardAdditiveModel
from tabpfn.models.transformer_make_model import EnsembleMeta, ShiftClassifier

ADDITIVE_MOTHERNET_PATH = "models_diff/additive_for_testing.cpkt"


def test_predict_basic_iris():
    pytest.skip("haven't checked in model checkpoints yet")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = ForwardAdditiveModel(path=ADDITIVE_MOTHERNET_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.55

from tabpfn.transformer_make_model import ShiftClassifier, EnsembleMeta, ForwardMLPModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pytest

MOTHERNET_PATH = "models_diff/prior_diff_real_checkpointcontinue_hidden_128_embed_dim_1024_decoder_nhid_2048_nlayer12_lr0003_n_0_epoch_on_exit.cpkt"
TABPFN_PATH = ""

@pytest.mark.parametrize("ensemble", [ShiftClassifier, EnsembleMeta, None])
@pytest.mark.parametrize("class_offset", [0, 4])
def test_basic_test_iris(ensemble, class_offset):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = ForwardMLPModel(path=MOTHERNET_PATH, device='cpu')
    if ensemble is not None:
        mothernet = ensemble(mothernet)
    mothernet.fit(X_train, y_train + class_offset)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test + class_offset) > 0.9
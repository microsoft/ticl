from tabpfn.transformer_make_model import ShiftClassifier, EnsembleMeta, ForwardMLPModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pytest

MOTHERNET_PATH = "models_diff/prior_diff_real_checkpointcontinue_hidden_128_embed_dim_1024_decoder_nhid_2048_nlayer12_lr0003_n_0_epoch_on_exit.cpkt"
MOTHERNET_L2_PATH = "models_diff/mothernet_128_decoder_2048_emsize_512_nlayers_12_steps_8192_bs_8ada_lr_3e-05_1_gpu_07_31_2023_23_18_33_epoch_780.cpkt"
MOTHERNET_LOW_RANK_PATH = "models_diff/mn_n1024_L2_W128_P512_1_gpu_08_03_2023_03_48_19_epoch_on_exit.cpkt"

@pytest.mark.parametrize("ensemble", [ShiftClassifier, EnsembleMeta, None])
@pytest.mark.parametrize("class_offset", [0, 4])
def test_basic_iris(ensemble, class_offset):
    if class_offset == 4 and ensemble in [None, ShiftClassifier]:
        raise pytest.skip("Skip this test because the ensemble is None and class_offset is 4")
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


def test_two_layers_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = ForwardMLPModel(path=MOTHERNET_L2_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9


def test_low_rank_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = ForwardMLPModel(path=MOTHERNET_LOW_RANK_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (38,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (38, 3)
    assert mothernet.score(X_test, y_test) > 0.9
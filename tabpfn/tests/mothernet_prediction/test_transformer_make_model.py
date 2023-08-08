from tabpfn.transformer_make_model import ShiftClassifier, EnsembleMeta, ForwardMLPModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MOTHERNET_PATH = "models_diff/prior_diff_real_checkpointcontinue_hidden_128_embed_dim_1024_decoder_nhid_2048_nlayer12_lr0003_n_0_epoch_on_exit.cpkt"
TABPFN_PATH = ""


def test_basic_test_iris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    mothernet = ForwardMLPModel(path=MOTHERNET_PATH, device='cpu')
    mothernet.fit(X_train, y_train)
    pred = mothernet.predict(X_test)
    assert pred.shape == (50,)
    pred_prob = mothernet.predict_proba(X_test)
    assert pred_prob.shape == (50, 3)
    assert mothernet.score(X_test, y_test) > 0.9
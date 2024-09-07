from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment',  '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1',
                    '--decoder-activation', 'relu', '--validate', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment',  '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1',
                          '--decoder-activation', 'relu',
                          '--save-every', '2', '--validate', 'False']


def get_model_path(results):
    return f"{results['base_path']}/models_diff/{results['model_string']}_epoch_{results['epoch']}.cpkt"


def check_predict_iris(clf, check_accuracy=False):
    # smoke test for predict, models aren't trained enough to check for accuracy
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    if check_accuracy:
        assert clf.score(X_test, y_test) > 0.9


def check_predict_moneyball(reg, check_score=False):
    # smoke test for predict, models aren't trained enough to check for accuracy
    data = fetch_openml("Moneyball")
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
    prep = make_column_transformer((OrdinalEncoder(), X_train.dtypes == "category"), remainder='passthrough')
    X_train_pre = prep.fit_transform(X_train)
    reg.fit(X_train_pre, y_train)
    y_pred = reg.predict(prep.transform(X_test))
    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1 or y_pred.shape[1] == 1
    if check_score:
        assert reg.score(X_test, y_test) > 0.9


def check_predict_linear(reg, check_score=False):
    # smoke test for predict, models aren't trained enough to check for accuracy
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 2))
    y = X @ rng.normal(size=(2,))
    reg.fit(X, y)
    y_pred = reg.predict(X)
    assert y_pred.shape[0] == X.shape[0]
    assert y_pred.ndim == 1 or y_pred.shape[1] == 1
    if check_score:
        assert reg.score(X, y) > 0.9

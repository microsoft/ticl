from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1',
                          '--save-every', '2']


def get_model_path(results):
    return f"{results['base_path']}/models_diff/{results['model_string']}_epoch_{results['epoch']}.cpkt"


def check_predict_iris(clf):
    # smoke test for predict, models aren't trained enough to check for accuracy
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]

import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.tabpfn import TabPFN
from mothernet.prediction import TabPFNClassifier

from mothernet.testing_utils import count_parameters, check_predict_iris

TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                          '--save-every', '2']


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['model_string'].startswith("tabpfn_AFalse_e128_E10_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(0.7096278667449951, rel=1e-5)


def test_train_tabpfn_num_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--num-features', '13'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert results['model'].encoder.weight.shape[1] == 13
    assert count_parameters(results['model']) == 568714
    assert results['loss'] == pytest.approx(0.6972318291664124, rel=1e-5)


def test_train_tabpfn_num_samples():
    # smoke test only since I'm too lazy to mock
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--n-samples', '35'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.6528608798980713, rel=1e-5)


def test_train_tabpfn_init_weights():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--init-method', 'kaiming-uniform'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.7984751462936401)


def test_train_tabpfn_init_weights_no_zero():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--init-method', 'kaiming-uniform', '--tabpfn-zero-weights', 'False'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.9158223867416382)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--multiclass-type', 'steps'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(1.4251887798309326)


def test_train_tabpfn_stepped_multiclass_steps3():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--multiclass-type', 'steps', '--multiclass-max-steps', '3'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(1.6232173442840576)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.688378632068634)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only', '--p-uninformative', '.9'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.6858838200569153)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only', '--max-fraction-uninformative', '2'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.6879596710205078)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn', '-C', '-E', '30', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                       'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0', '--reduce-lr-on-spike',
                        'True', '-B', tmpdir, '--prior-type', 'bag_boolean'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.687474250793457)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--add-uninformative-features', 'True'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.691472589969635)

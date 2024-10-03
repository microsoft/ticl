import tempfile

import lightning as L
import pytest

from ticl.fit_model import main
from ticl.models.tabpfn import TabPFN
from ticl.prediction import TabPFNClassifier

from ticl.testing_utils import count_parameters, check_predict_iris

TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                    'testing_experiment', '--train-mixed-precision', 'False', '--validate', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                          'testing_experiment', '--train-mixed-precision', 'False',
                          '--save-every', '2', '--validate', 'False']


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['model_string'].startswith("tabpfn_AFalse_e128_E10_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(0.7048388719558716, rel=1e-4)


def test_train_tabpfn_num_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--num-features', '13'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert results['model'].encoder.weight.shape[1] == 13
    assert count_parameters(results['model']) == 568714
    assert results['loss'] == pytest.approx(0.6869455575942993, rel=1e-5)


def test_train_tabpfn_num_samples():
    # smoke test only since I'm too lazy to mock
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--n-samples', '35'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.7099079489707947, rel=1e-5)


def test_train_tabpfn_init_weights():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--init-method', 'kaiming-uniform'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(2.239567518234253)


def test_train_tabpfn_init_weights_no_zero():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--init-method', 'kaiming-uniform', '--tabpfn-zero-weights', 'False'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(2.207158088684082)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--multiclass-type', 'steps'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(1.1241533756256104)


def test_train_tabpfn_stepped_multiclass_steps3():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--multiclass-type', 'steps', '--multiclass-max-steps', '3'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.9533907771110535)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.7127256989479065)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only', '--p-uninformative', '.9'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.7100008130073547)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--prior-type', 'boolean_only', '--max-fraction-uninformative', '2'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.7263631820678711)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn', '-C', '-E', '30', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                       'testing_experiment', '--train-mixed-precision', 'False', '--min-lr', '0', '--reduce-lr-on-spike',
                        'True', '-B', tmpdir, '--prior-type', 'bag_boolean', '--validate', 'False', '--seed-everything', 'False'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.7003629207611084)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabpfn'] + TESTING_DEFAULTS + ['-B', tmpdir, '--add-uninformative-features', 'True'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.696788489818573)

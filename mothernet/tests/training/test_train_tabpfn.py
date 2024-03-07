import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.tabpfn import TabPFN
from mothernet.prediction import TabPFNClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert results['loss'] == pytest.approx(1.6330838203430176, rel=1e-5)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_num_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--num-features', '13'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert results['model'].encoder.weight.shape[1] == 13
    assert count_parameters(results['model']) == 568714
    assert results['loss'] == pytest.approx(0.7162004709243774, rel=1e-5)


def test_train_tabpfn_num_samples():
    # smoke test only since I'm too lazy to mock
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--n-samples', '35'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.6996666193008423, rel=1e-5)


def test_train_tabpfn_init_weights():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--init-method', 'kaiming-uniform'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(1.1084306240081787)


def test_train_tabpfn_init_weights_no_zero():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--init-method', 'kaiming-uniform', '--tabpfn-zero-weights', 'False'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(1.2844524383544922)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--multiclass-type', 'steps'])
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(0.5968285799026489)


def test_train_tabpfn_stepped_multiclass_steps3():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'tabpfn', '--multiclass-type', 'steps', '--multiclass-max-steps', '3'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.6931300163269043)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.723449170589447)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-p-uninformative', '.9'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.713407039642334)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-max-fraction-uninformative', '2'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.7003865242004395)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '30', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                       'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike',
                        'True', '-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'bag_boolean'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(0.6881492137908936)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--add-uninformative-features', 'True'])
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)
    assert results['loss'] == pytest.approx(1.12027883529663092)

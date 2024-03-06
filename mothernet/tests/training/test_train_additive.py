import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.mothernet_additive import MotherNetAdditive
from mothernet.prediction import MotherNetAdditiveClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris, get_model_path


def test_train_additive_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9690634
    assert results['loss'] == pytest.approx(0.7711865901947021, rel=1e-5)


def test_train_additive_class_tokens():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_tokens'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(2.492729902267456, rel=1e-5)


def test_train_additive_class_average():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_average'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(2.0791170597076416, rel=1e-5)


def test_train_additive_class_average_input_layer_norm():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_average', '--input-layer-norm', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2205697
    assert results['loss'] == pytest.approx(6.360681056976318, rel=1e-5)
                                            

def test_train_additive_input_bin_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7029958367347717, rel=1e-5)


def test_train_additive_special_token_simple():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'special_token_simple'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9624586
    assert results['loss'] == pytest.approx(3.113452911376953, rel=1e-5)


def test_train_additive_input_bin_embedding_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True', '--bin-embedding-rank', '8'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 8)
    assert count_parameters(results['model']) == 8975018
    assert results['loss'] == pytest.approx(0.7063571810722351, rel=1e-5)


def test_train_additive_input_bin_embedding_linear():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'linear'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7057666182518005, rel=1e-5)


def test_train_additive_factorized_output():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1649994
    assert results['loss'] == pytest.approx(2.0514190196990967, rel=1e-5)


def test_train_additive_factorized_output_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--output-rank', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64, 10)
    assert count_parameters(results['model']) == 1487514
    assert results['loss'] == pytest.approx(1.0063064098358154, rel=1e-5)


def test_train_additive_class_average_factorized():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True',
                                                 '--output-rank', '4', '--decoder-type', 'class_average'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64)
    assert count_parameters(results['model']) == 1419034
    assert results['loss'] == pytest.approx(0.8343020081520081, rel=1e-5)


def test_train_additive_factorized_in_and_out():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--input-bin-embedding', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1038090
    assert results['loss'] == pytest.approx(3.638622283935547, rel=1e-5)
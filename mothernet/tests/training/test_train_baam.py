import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_additive_mothernet import BiAttentionMotherNetAdditive
from mothernet.prediction.mothernet_additive import MotherNetAdditiveClassifier

from mothernet.testing_utils import count_parameters, check_predict_iris, get_model_path

TESTING_DEFAULTS = ['baam', '-C', '-E', '8', '-n', '1', '-A', 'False', '-e', '16', '-N', '2', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--num-features', '20', '--n-samples', '200',
                    '--decoder-activation', 'relu', '--save-every', '8']


def test_train_baam_shape_attention():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--factorized-output', 'True',
                                           '--shape-attention', 'True', '--shape-attention-heads', '2', '--n-shape-functions', '16', '--shape-init', 'constant',
                                           '--output-rank', '8'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert results['model_string'].startswith("baam_AFalse_decoderactivationrelu_e16_E8_factorizedoutputTrue_nsamples200_nshapefunctions16_N2_numfeatures20"
                                              "_n1_outputrank8_shapeattentionTrue_shapeattentionheads2_tFalse_cpu_")
    assert count_parameters(results['model']) == 24340
    assert results['loss'] == pytest.approx(0.6722398400306702, rel=1e-5)


def test_train_baam_nbins():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--n-bins', '512'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
        assert clf.w_.shape == (4, 512, 3)

    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert results['model_string'].startswith("baam_AFalse_decoderactivationrelu_e16_E8_nbins512_nsamples200_N2_numfeatures20_n1_tFalse_cpu_")
    assert count_parameters(results['model']) == 288640
    assert results['model'].decoder.mlp[2].weight.shape == (512, 512)
    assert results['loss'] == pytest.approx(1.5397337675094604, rel=1e-5)


def test_train_baam_no_shape_attention():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert count_parameters(results['model']) == 51648
    assert results['loss'] == pytest.approx(0.71629798412323, rel=1e-5)


def test_train_baam_marginal_residual_no_learning():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--marginal-residual', 'True', '-l', '0', '--shape-init', 'zero', '-E', '1', '--save-every', '1'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf, check_accuracy=True)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert count_parameters(results['model']) == 55744
    assert results['loss'] == pytest.approx(1.1165376901626587, rel=1e-5)


def test_train_baam_fourier_features():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--fourier-features', '32'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert results['model'].encoder.weight.shape == (16, 97)
    assert count_parameters(results['model']) == 52176
    assert results['loss'] == pytest.approx(0.6272056102752686, rel=1e-5)


def test_train_baam_input_layer_norm():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--input-layer-norm', 'True'])
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert count_parameters(results['model']) == 51776
    assert results['loss'] == pytest.approx(0.6908526420593262, rel=1e-5)


def test_train_baam_class_average_no_y_encoder():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'class_average', '--y-encoder', 'None'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert results['model'].decoder_type == "class_average"
    assert count_parameters(results['model']) == 51472
    assert results['model'].y_encoder is None
    assert results['model'].decoder.mlp[0].in_features == 16
    assert results['loss'] == pytest.approx(1.2202619314193726)

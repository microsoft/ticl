import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_additive_mothernet import BiAttentionMotherNetAdditive

from mothernet.testing_utils import count_parameters

TESTING_DEFAULTS = ['baam', '-C', '-E', '8', '-n', '1', '-A', 'False', '-e', '16', '-N', '2', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--num-features', '20', '--n-samples', '200']


def test_train_baam_shape_attention():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--factorized-output', 'True',
                                           '--shape-attention', 'True', '--shape-attention-heads', '2', '--n-shape-functions', '16', '--shape-init', 'constant',
                                           '--output-rank', '8'])
        # clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        # check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert results['model_string'].startswith("baam_AFalse_e16_E8_factorizedoutputTrue_nsamples200_nshapefunctions16_N2_numfeatures20"
                                              "_n1_outputrank8_shapeattentionTrue_shapeattentionheads2_tFalse_cpu_03_1")
    assert count_parameters(results['model']) == 62740
    assert results['loss'] == pytest.approx(1.6953184604644775, rel=1e-5)


def test_train_baam_no_shape_attention():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
        # clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        # check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert count_parameters(results['model']) == 176064
    assert results['loss'] == pytest.approx(0.7351612448692322, rel=1e-5)


def test_train_baam_input_layer_norm():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--input-layer-norm', 'True'])
        # clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        # check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionMotherNetAdditive)
    assert count_parameters(results['model']) == 176192
    assert results['loss'] == pytest.approx(1.1924283504486084, rel=1e-5)

import tempfile

import lightning as L
import pytest

from ticl.fit_model import main
from ticl.models.biattention_tabpfn import BiAttentionTabPFN

from ticl.prediction import TabPFNClassifier
from ticl.testing_utils import count_parameters, check_predict_iris


TESTING_DEFAULTS = ['batabpfn', '-C', '-E', '8', '-n', '1', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                    'testing_experiment',  '--train-mixed-precision', 'False', '--num-features', '20', '--n-samples', '200', '--save-every', '8', '--validate', 'False']


def test_train_batabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--pad-zeros', 'True', '--input-embedding', 'linear'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)

    assert isinstance(results['model'], BiAttentionTabPFN)
    assert results['model_string'].startswith("batabpfn_AFalse_e4_E8_inputembeddinglinear_nsamples200_N2_numfeatures20_n1_padzerosTrue_tFalse_cpu")
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.205095052719116, rel=1e-5)


def test_train_batabpfn_no_padding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--input-embedding', 'linear'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.2096073627471924, rel=1e-5)


def test_train_batabpfn_random_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment',  '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--input-embedding', 'random', '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(1.2611584514379501, rel=1e-5)


def test_train_batabpfn_fourier_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment',  '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert results['model_string'].startswith("batabpfn_AFalse_e4_E8_nsamples200_N2_numfeatures20_n4_tFalse_cpu")
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.9176976233720779, rel=1e-5)


def test_train_batabpfn_fourier_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment',  '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.8441332280635834, rel=1e-5)


def test_train_batabpfn_linear_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment',  '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--input-embedding', 'linear', '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(0.6607210636138916, rel=1e-5)


def test_train_batabpfn_random_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment',  '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--input-embedding', 'random', '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(0.972737699747085, rel=1e-5)

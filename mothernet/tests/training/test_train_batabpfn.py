import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_tabpfn import BiAttentionTabPFN

from mothernet.testing_utils import count_parameters

TESTING_DEFAULTS = ['batabpfn', '-C', '-E', '8', '-n', '1', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--num-features', '20', '--n-samples', '200']


def test_train_batabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert results['model_string'].startswith("batabpfn_AFalse_e4_E8_nsamples200_N2_numfeatures20_n1_tFalse_cpu_03_13_2024_15_33_46")
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.108328342437744, rel=1e-5)


def test_train_batabpfn_no_padding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--pad-zeros', 'False'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.103300094604492, rel=1e-5)


def test_train_batabpfn_random_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--pad-zeros', 'False', '--input-embedding', 'random', '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(0.9694245010614395, rel=1e-5)


def test_train_batabpfn_fourier_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--pad-zeros', 'False', '--input-embedding', 'fourier', '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.7749457061290741, rel=1e-5)


def test_train_batabpfn_fourier_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--pad-zeros', 'False', '--input-embedding', 'fourier', '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.7936984598636627, rel=1e-5)


def test_train_batabpfn_linear_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--pad-zeros', 'False', '--input-embedding', 'linear', '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(0.849937304854393, rel=1e-5)


def test_train_batabpfn_random_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['batabpfn', '-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '--pad-zeros', 'False', '--input-embedding', 'random', '--num-features', '20'],
                       extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(1.1738452464342117, rel=1e-5)
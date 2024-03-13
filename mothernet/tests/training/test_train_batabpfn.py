import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_tabpfn import BiAttentionTabPFN

from mothernet.testing_utils import count_parameters

TESTING_DEFAULTS = ['-C', '-E', '8', '-n', '1', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--num-features', '20', '--n-samples', '200']


def test_train_batabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'batabpfn'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.108328342437744, rel=1e-5)


def test_train_batabpfn_no_padding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'batabpfn', '--pad-zeros', 'False'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(2.103300094604492, rel=1e-5)


def test_train_batabpfn_random_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '-m', 'batabpfn', '--pad-zeros', 'False', '--input-embedding', 'random', '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 870
    assert results['loss'] == pytest.approx(0.9694245010614395, rel=1e-5)


def test_train_batabpfn_fourier_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '-m', 'batabpfn', '--pad-zeros', 'False', '--input-embedding', 'fourier', '--num-features', '20'])
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.7749457061290741, rel=1e-5)


def test_train_batabpfn_fourier_features_nans():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_config = {'prior': {'classification': {'nan_prob_no_reason': 0.5}}}
        results = main(['-C', '-E', '8', '-n', '4', '-A', 'False', '-e', '4', '-N', '2', '--experiment',
                        'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False',
                        '--n-samples', '200', '-B', tmpdir, '-m', 'batabpfn', '--pad-zeros', 'False', '--input-embedding', 'fourier', '--num-features', '20'], extra_config)
    assert isinstance(results['model'], BiAttentionTabPFN)
    assert count_parameters(results['model']) == 886
    assert results['loss'] == pytest.approx(0.7749457061290741, rel=1e-5)

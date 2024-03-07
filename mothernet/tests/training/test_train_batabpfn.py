import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_tabpfn import BiAttentionTabPFN
# from mothernet.prediction import TabPFNClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris


def test_train_batabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'batabpfn'])
        # clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        # check_predict_iris(clf)
    assert results['loss'] == pytest.approx(1.6330838203430176, rel=1e-5)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], BiAttentionTabPFN)


def test_train_batabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'batabpfn', '--multiclass-type', 'steps'])
    assert results['loss'] == pytest.approx(0.5968285799026489)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], BiAttentionTabPFN)


def test_train_batabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'batabpfn', '--add-uninformative-features', 'True'])
    assert results['loss'] == pytest.approx(1.12027883529663092)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], BiAttentionTabPFN)
    
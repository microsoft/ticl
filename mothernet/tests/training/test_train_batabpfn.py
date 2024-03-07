import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.biattention_tabpfn import BiAttentionTabPFN
# from mothernet.prediction import TabPFNClassifier

from mothernet.testing_utils import count_parameters

TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '4', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '4', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False', '-L', '1',
                          '--save-every', '2']


def test_train_batabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'batabpfn'])
        # clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        # check_predict_iris(clf)
    assert results['loss'] == pytest.approx(1.6330838203430176, rel=1e-5)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], BiAttentionTabPFN)
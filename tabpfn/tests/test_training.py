import tempfile
import pytest

from tabpfn.fit_model import main
from tabpfn.transformer_make_model import TransformerModelMakeMLP
from tabpfn.transformer import TransformerModel
from tabpfn.perceiver import TabPerceiver


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', '-e', '128', '-N', '4', '-P', '64']


def test_train_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_double_embedding():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_special_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S'])
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_two_hidden_layers():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L' '2'])
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_low_rank():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W' '16'])
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_tabpfn():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert isinstance(model, TransformerModel)

@pytest.mark.skip(reason="Perceiver is not yet implemented after refactor")
def test_train_perceiver():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    assert isinstance(model, TabPerceiver)
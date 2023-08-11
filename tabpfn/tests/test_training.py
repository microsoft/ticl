import tempfile
import pytest

from tabpfn.fit_model import main
from tabpfn.transformer_make_model import TransformerModelMakeMLP
from tabpfn.transformer import TransformerModel
from tabpfn.perceiver import TabPerceiver
import lightning as L


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', '-e', '128', '-N', '4', '-P', '64']


def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert loss == 2.350903034210205
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_double_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert loss == 2.313856601715088
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S'])
    assert loss == 3.088744878768921
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L' '2'])
    assert loss == 2.2727622985839844
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W' '16'])
    assert loss == 2.243964910507202
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_tabpfn():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert loss == 2.306408643722534
    assert isinstance(model, TransformerModel)

@pytest.mark.skip(reason="Perceiver is not yet implemented after refactor")
def test_train_perceiver():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    assert loss == 2.350903034210205
    assert isinstance(model, TabPerceiver)
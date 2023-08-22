import tempfile
import pytest

from tabpfn.fit_model import main
from tabpfn.transformer_make_model import TransformerModelMakeMLP
from tabpfn.transformer import TransformerModel
from tabpfn.perceiver import TabPerceiver
import lightning as L

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment', 'testing_experiment']


def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert loss == 2.4132816791534424
    assert count_parameters(model) == 1544650
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_double_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert loss == 2.2314932346343994
    assert count_parameters(model) == 1775818
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S'])
    assert loss == 2.3119993209838867
    assert count_parameters(model) == 1544650
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L' '2'])
    assert loss == 2.3295023441314697
    assert count_parameters(model) == 2081290
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W' '16'])
    assert loss == 2.3065733909606934
    assert count_parameters(model) == 926474
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_tabpfn():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert loss == 2.3345985412597656
    assert count_parameters(model) == 579850
    assert isinstance(model, TransformerModel)

def test_train_perceiver():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    assert loss == 2.1738672256469727
    assert count_parameters(model) == 2930634
    assert isinstance(model, TabPerceiver)

def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert loss == 2.1443655490875244
    assert count_parameters(model) == 3467274
    assert isinstance(model, TabPerceiver)

def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16'])
    assert loss == 2.1738672256469727
    assert count_parameters(model) == 2930634
    assert isinstance(model, TabPerceiver)
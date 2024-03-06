import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
# from tabpfn.fit_tabpfn import main as tabpfn_main
from mothernet.models.perceiver import TabPerceiver

from mothernet.testing_utils import TESTING_DEFAULTS, count_parameters


def test_train_perceiver_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    model = results['model']
    assert isinstance(model, TabPerceiver)
    assert model.ff_dropout == 0
    assert model.input_dim == 128
    assert len(model.layers) == 4
    assert model.latents.shape == (512, 128)
    assert model.decoder.hidden_size == 128
    assert model.decoder.emsize == 128
    assert count_parameters(model.decoder) == 1000394
    assert count_parameters(model.encoder) == 12928
    assert count_parameters(model.layers) == 664576
    assert count_parameters(model) == 1744842
    assert results['loss'] == pytest.approx(1.0585685968399048)


def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert isinstance(results['model'], TabPerceiver)
    assert count_parameters(results['model']) == 2281482
    assert results['loss'] == pytest.approx(0.795587420463562)


def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16', '--low-rank-weights', 'True'])
    assert isinstance(results['model'], TabPerceiver)
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 2314
    assert count_parameters(results['model']) == 1126666
    assert results['loss'] == pytest.approx(0.6930023431777954, rel=1e-5)

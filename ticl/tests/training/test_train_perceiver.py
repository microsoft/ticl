import tempfile

import lightning as L
import pytest

from ticl.fit_model import main
# from tabpfn.fit_tabpfn import main as tabpfn_main
from ticl.models.perceiver import TabPerceiver

from ticl.testing_utils import TESTING_DEFAULTS, count_parameters


def test_train_perceiver_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(["perceiver", '-B', tmpdir, '-D', 'output_attention'] + TESTING_DEFAULTS)
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
    assert results['loss'] == pytest.approx(0.8764909505844116)
    assert results['model_string'].startswith("perceiver_AFalse_decoderactivationrelu_d128_H128_e128_E10_rFalse_N4_n1_P64_L1_tFalse_cpu")


def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(["perceiver", '-B', tmpdir, '-D', 'output_attention'] + TESTING_DEFAULTS + ['-L', '2'])
    assert isinstance(results['model'], TabPerceiver)
    assert count_parameters(results['model']) == 2281482
    assert results['loss'] == pytest.approx(0.7871301770210266)


def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(["perceiver", '-B', tmpdir, '-D', 'output_attention'] + TESTING_DEFAULTS + ['-W', '16', '--low-rank-weights', 'True'])
    assert isinstance(results['model'], TabPerceiver)
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 2314
    assert count_parameters(results['model']) == 1126666
    assert results['loss'] == pytest.approx(0.7207471132278442, rel=1e-5)

import tempfile

import lightning as L
import pytest
import torch

from mothernet.fit_model import main
from mothernet.models.mothernet_additive import MotherNetAdditive
from mothernet.prediction import MotherNetAdditiveClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris, get_model_path

TESTING_DEFAULTS_ADDITIVE = ['additive'] + TESTING_DEFAULTS
TESTING_DEFAULTS_SHORT_ADDITIVE = ['additive'] + TESTING_DEFAULTS_SHORT


def test_train_additive_old_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--decoder-type', 'output_attention'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model_string'].startswith("additive_AFalse_decoderactivationrelu_d128_H128_Doutput_attention_e128_E10_rFalse_N4_n1_P64_L1_tFalse_cpu_")
    assert count_parameters(results['model']) == 9690634
    assert results['loss'] == pytest.approx(0.8093194961547852, rel=1e-5)


def test_train_additive_nbins():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--n-bins', '128'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
        assert clf.w_.shape == (4, 128, 3)

    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model_string'].startswith("additive_AFalse_decoderactivationrelu_d128_H128_e128_E10_rFalse_nbins128_N4_n1_P64_L1_tFalse_cpu")
    assert count_parameters(results['model']) == 3837697
    assert results['model'].decoder.mlp[2].weight.shape[0] == 12801
    assert results['loss'] == pytest.approx(1.582958459854126, rel=1e-5)


def test_train_additive_validation():
    # FIXME not actually testing that validation worked
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--validate', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897


def test_train_additive_class_average_shape_attention():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (32, 64)
    assert results['model'].decoder.shape_function_keys.shape == (32, 4)
    assert count_parameters(results['model']) == 1420954
    assert results['loss'] == pytest.approx(0.7705700993537903, rel=1e-5)


def test_train_additive_class_average_multihead_shape_attention():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True',
                                                          '--shape-attention-heads', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (32, 64)
    assert len(results['model'].decoder.shape_function_keys) == 4  # number of attention heads
    assert results['model'].decoder.shape_function_keys[0].shape == (32, 4)
    assert results['model'].decoder.shape_functions.std().item() == pytest.approx(1, rel=2e-2)
    assert count_parameters(results['model']) == 1421406
    assert results['loss'] == pytest.approx(2.2196261882781982, rel=1e-5)


def test_train_additive_class_average_multihead_shape_attention_init():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True',
                                                          '--shape-attention-heads', '4', '--shape-init', 'inverse'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (32, 64)
    assert len(results['model'].decoder.shape_function_keys) == 4  # number of attention heads
    assert results['model'].decoder.shape_function_keys[0].shape == (32, 4)
    assert results['model'].decoder.shape_functions.std().item() == pytest.approx(1/(32 * 64), rel=1e-2, abs=4e-4)
    assert count_parameters(results['model']) == 1421406
    assert results['loss'] == pytest.approx(1.9031586647033691, rel=1e-5)


def test_train_additive_class_average_multihead_shape_attention_shape_functions8():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True',
                                                          '--shape-attention-heads', '4', '--n-shape-functions', '8'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (8, 64)
    assert len(results['model'].decoder.shape_function_keys) == 4  # number of attention heads
    assert results['model'].decoder.shape_function_keys[0].shape == (8, 4)
    assert count_parameters(results['model']) == 1419486
    assert results['loss'] == pytest.approx(1.2250710725784302, rel=1e-5)


def test_train_additive_class_tokens():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--decoder-type', 'class_tokens'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(0.9419207572937012, rel=1e-5)


def test_train_additive_class_average():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--decoder-type', 'class_average'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(0.8486008644104004, rel=1e-5)


def test_train_additive_class_average_input_layer_norm():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_ADDITIVE + ['-B', tmpdir, '--decoder-type', 'class_average', '--input-layer-norm', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2205697
    assert results['loss'] == pytest.approx(0.7265084981918335, rel=1e-5)


def test_train_additive_input_bin_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--input-bin-embedding', 'True', '--decoder-type', 'output_attention'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.8084635734558105, rel=1e-5)


def test_train_additive_variable_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--num-features', '10'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder[1].weight.shape == (128, 640)
    assert count_parameters(results['model']) == 712577
    assert results['loss'] == pytest.approx(0.7357390522956848, rel=1e-5)


def test_train_additive_special_token_simple():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--decoder-type', 'special_token_simple'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9624586
    assert results['loss'] == pytest.approx(1.0065537691116333, rel=1e-5)


def test_train_additive_input_bin_embedding_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--input-bin-embedding', 'True', '--bin-embedding-rank', '8',
                                                          '--decoder-type', 'output_attention'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 8)
    assert count_parameters(results['model']) == 8975018
    assert results['loss'] == pytest.approx(0.692044734954834, rel=1e-5)


def test_train_additive_input_bin_embedding_linear():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--input-bin-embedding', 'linear',
                                                          '--decoder-type', 'output_attention'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7215449810028076, rel=1e-5)


def test_train_additive_factorized_output():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--decoder-type', 'output_attention'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1649994
    assert results['loss'] == pytest.approx(1.0714590549468994, rel=1e-5)


def test_train_additive_factorized_output_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True', '--output-rank', '4',
                                                          '--decoder-type', 'output_attention'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64, 10)
    assert count_parameters(results['model']) == 1487514
    assert results['loss'] == pytest.approx(0.8312861919403076, rel=1e-5)


def test_train_additive_class_average_factorized():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True',
                                                          '--output-rank', '4', '--decoder-type', 'class_average'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64)
    assert count_parameters(results['model']) == 1419034
    assert results['loss'] == pytest.approx(0.9980924129486084, rel=1e-5)


def test_train_additive_factorized_in_and_out():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT_ADDITIVE + ['-B', tmpdir, '--factorized-output', 'True', '--input-bin-embedding', 'True',
                                                          '--decoder-type', 'output_attention'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1038090
    assert results['loss'] == pytest.approx(1.255028247833252, rel=1e-5)


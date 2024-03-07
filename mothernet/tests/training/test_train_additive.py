import tempfile

import lightning as L
import pytest

from mothernet.fit_model import main
from mothernet.models.mothernet_additive import MotherNetAdditive
from mothernet.prediction import MotherNetAdditiveClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris, get_model_path


def test_train_additive_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9690634
    assert results['loss'] == pytest.approx(1.205582857131958, rel=1e-5)


def test_train_additive_class_average_shape_attention():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True',
                                                 '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (32, 64)
    assert results['model'].decoder.shape_function_keys.shape == (32, 4)
    assert count_parameters(results['model']) == 1420954
    assert results['loss'] == pytest.approx(1.395249366760254, rel=1e-5)


def test_train_additive_class_average_multihead_shape_attention():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True',
                                                 '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True',
                                                 '--shape-attention-heads', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (32, 64)
    assert len(results['model'].decoder.shape_function_keys) == 4  # number of attention heads
    assert results['model'].decoder.shape_function_keys[0].shape == (32, 4)
    assert count_parameters(results['model']) == 1421406
    assert results['loss'] == pytest.approx(0.8446271419525146, rel=1e-5)


def test_train_additive_class_average_multihead_shape_attention_shape_functions8():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True',
                                                 '--output-rank', '4', '--decoder-type', 'class_average', '--shape-attention', 'True',
                                                 '--shape-attention-heads', '4', '--n-shape-functions', '8'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.shape_functions.shape == (8, 64)
    assert len(results['model'].decoder.shape_function_keys) == 4  # number of attention heads
    assert results['model'].decoder.shape_function_keys[0].shape == (8, 4)
    assert count_parameters(results['model']) == 1419486
    assert results['loss'] == pytest.approx(1.8396743535995483, rel=1e-5)


def test_train_additive_class_tokens():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_tokens'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(2.564612865447998, rel=1e-5)


def test_train_additive_class_average():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_average'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2192897
    assert results['loss'] == pytest.approx(2.1230266094207764, rel=1e-5)


def test_train_additive_class_average_input_layer_norm():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_average', '--input-layer-norm', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 2205697
    assert results['loss'] == pytest.approx(6.593766212463379, rel=1e-5)
                                            

def test_train_additive_input_bin_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7059590816497803, rel=1e-5)


def test_train_additive_special_token_simple():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'special_token_simple'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9624586
    assert results['loss'] == pytest.approx(1.060562252998352, rel=1e-5)


def test_train_additive_input_bin_embedding_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True', '--bin-embedding-rank', '8'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 8)
    assert count_parameters(results['model']) == 8975018
    assert results['loss'] == pytest.approx(0.7024991512298584, rel=1e-5)


def test_train_additive_input_bin_embedding_linear():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'linear'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7775996923446655, rel=1e-5)


def test_train_additive_factorized_output():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1649994
    assert results['loss'] == pytest.approx(2.0079495906829834, rel=1e-5)


def test_train_additive_factorized_output_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--output-rank', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64, 10)
    assert count_parameters(results['model']) == 1487514
    assert results['loss'] == pytest.approx(1.3647632598876953, rel=1e-5)


def test_train_additive_class_average_factorized():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True',
                                                 '--output-rank', '4', '--decoder-type', 'class_average'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64)
    assert count_parameters(results['model']) == 1419034
    assert results['loss'] == pytest.approx(0.8229849338531494, rel=1e-5)


def test_train_additive_factorized_in_and_out():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--input-bin-embedding', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1038090
    assert results['loss'] == pytest.approx(1.3997379541397095, rel=1e-5)
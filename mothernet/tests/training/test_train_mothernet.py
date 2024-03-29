
import tempfile

import lightning as L
import numpy as np
import pytest

from mothernet.fit_model import main

from mothernet.models.mothernet import MotherNet
from mothernet.config_utils import compare_dicts
from mothernet.prediction import MotherNetClassifier

from mothernet.testing_utils import TESTING_DEFAULTS, TESTING_DEFAULTS_SHORT, count_parameters, check_predict_iris, get_model_path

DEFAULT_LOSS = pytest.approx(0.696098268032074)

TESTING_DEFAULTS_MOTHERNET = ['mothernet'] + TESTING_DEFAULTS
TESTING_DEFAULTS_MOTHERNET_SHORT = ['mothernet'] + TESTING_DEFAULTS_SHORT


def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['loss'] == DEFAULT_LOSS
    assert results['model_string'].startswith("mn_AFalse_decoderactivationrelu_d128_H128_e128_E10_rFalse_N4_n1_P64_L1_tFalse_cpu_")
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert count_parameters(results['model'].decoder) == 1000394


def test_train_gelu_decoder():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '--decoder-activation', 'gelu'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['loss'] == 0.6897920370101929
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert count_parameters(results['model'].decoder) == 1000394


def test_train_synetune():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['--st_checkpoint_dir', tmpdir])
        assert results['epoch'] == 10
        assert results['loss'] == DEFAULT_LOSS
        assert count_parameters(results['model']) == 1544650
        assert isinstance(results['model'], MotherNet)
        results = main(TESTING_DEFAULTS_MOTHERNET + ['--st_checkpoint_dir', tmpdir])
        # that we reloaded the model means we incidentally counted up to 11
        assert results['epoch'] == 11


def test_train_reload():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET_SHORT + ['-B', tmpdir, '--save-every', '1'])
        prev_file_name = f'{results["base_path"]}/models_diff/{results["model_string"]}_epoch_2.cpkt'
        assert results['epoch'] == 2
        assert results['loss'] == pytest.approx(0.7084088921546936)
        assert count_parameters(results['model']) == 1544650
        # "continue" training - will stop immediately since we already reached max epochs
        results_new = main(TESTING_DEFAULTS_MOTHERNET_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-c', '-R'])
        # epoch 3 didn't actually happen, but going through the training loop raises the counter by one...
        assert results_new['epoch'] == 3
        assert results_new['loss'] == np.inf
        assert results_new['base_path'] == results['base_path']
        assert results_new['model_string'].startswith(results['model_string'][:-20])
        ignored_configs = ['warm_start_from', 'continue_run']
        assert results_new['config']['orchestration']['warm_start_from'].split("/")[-1].startswith(results['model_string'])
        assert results_new['config']['orchestration']['continue_run']
        for k, v in results['config'].items():
            if k not in ignored_configs:
                if isinstance(v, dict):
                    assert compare_dicts(results_new['config'][k], v, return_bool=True, skip=ignored_configs)
                else:
                    assert results_new['config'][k] == v
        # strict loading should fail if we change model arch
        with pytest.raises(RuntimeError):
            main(TESTING_DEFAULTS_MOTHERNET_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-L', '2'])

        # strict loading should work if we change learning rate
        results = main(TESTING_DEFAULTS_MOTHERNET_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-l', '1e-3'])
        assert results['epoch'] == 2
        assert results['config']['optimizer']['learning_rate'] == 1e-3

        # non-strict loading should allow changing architecture
        results = main(TESTING_DEFAULTS_MOTHERNET_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-L', '2'])
        assert results['epoch'] == 2
        assert results['config']['mothernet']['predicted_hidden_layers'] == 2


def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-D', 'special_token'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['loss'] == pytest.approx(0.6808353066444397)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "special_token"
    assert count_parameters(results['model'].decoder) == 1000266
    assert results['model'].token_embedding.shape == (1, 1, 128)


def test_train_class_tokens():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-D', 'class_tokens'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "class_tokens"
    assert count_parameters(results['model']) == 1625930
    assert count_parameters(results['model'].decoder) == 1081674
    assert results['model'].decoder.mlp[0].in_features == 1280
    assert results['loss'] == pytest.approx(1.974905014038086)


def test_train_class_average():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-D', 'class_average'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "class_average"
    assert count_parameters(results['model']) == 1625930
    assert count_parameters(results['model'].decoder) == 1081674
    assert results['model'].decoder.mlp[0].in_features == 1280
    assert results['loss'] == pytest.approx(0.7590433359146118)


def test_train_simple_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-D', 'special_token_simple'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "special_token_simple"
    assert count_parameters(results['model']) == 1478602
    assert count_parameters(results['model'].decoder) == 934218
    assert results['model'].token_embedding.shape == (1, 1, 128)
    assert results['loss'] == pytest.approx( 4.234837532043457)


def test_train_average_decoder():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-D', 'average'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "average"
    assert count_parameters(results['model']) == 1478474
    assert count_parameters(results['model'].decoder) == 934218
    assert results['loss'] == pytest.approx(1.298040509223938)


def test_train_reduce_on_spike():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '--reduce-lr-on-spike', 'True'])
    assert results['loss'] == DEFAULT_LOSS
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)


def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-L', '2'])
    assert results['loss'] == pytest.approx(0.6612737774848938)
    assert count_parameters(results['model']) == 2081290
    assert isinstance(results['model'], MotherNet)


def test_train_two_decoder_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-T', '2'])
    assert isinstance(results['model'], MotherNet)
    assert count_parameters(results['model']) == 1561162
    assert results['loss'] == pytest.approx(0.9806529879570007)


def test_train_low_rank_ignored():
    # it boolean flag is not set, -W is ignored for easier hyperparameter search
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_MOTHERNET + ['-B', tmpdir, '-W', '16', '--low-rank-weights', 'False'])
    assert results['loss'] == DEFAULT_LOSS
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)


def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['mothernet', '-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128',
                        '--experiment', 'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',
                        '--reduce-lr-on-spike', 'True', '-B', tmpdir, '-W', '16', '--low-rank-weights', 'True', '-L', '2',
                        '--decoder-activation', 'relu'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['model_string'].startswith("mn_AFalse_decoderactivationrelu_d128_H128_e128_E10_minlr0_N4_n1_P64_reducelronspikeTrue_tFalse_W16_cpu_")
    assert count_parameters(results['model']) == 1067850
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 3402
    # suspiciously low tolerance here
    assert results['loss'] == pytest.approx(0.6916791796684265, rel=1e-4)
    assert isinstance(results['model'], MotherNet)

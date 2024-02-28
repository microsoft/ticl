import tempfile

import lightning as L
import numpy as np
import pytest

from mothernet.fit_model import main
# from tabpfn.fit_tabpfn import main as tabpfn_main
from mothernet.models.mothernet_additive import MotherNetAdditive
from mothernet.models.perceiver import TabPerceiver
from mothernet.models.tabpfn import TabPFN
from mothernet.models.mothernet import MotherNet
from mothernet.config_utils import compare_dicts
from mothernet.prediction import MotherNetClassifier, TabPFNClassifier, MotherNetAdditiveClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']

DEFAULT_LOSS = pytest.approx(1.0794482231140137)


def get_model_path(results):
    return f"{results['base_path']}/models_diff/{results['model_string']}_epoch_{results['epoch']}.cpkt"


def check_predict_iris(clf):
    # smoke test for predict, models aren't trained enough to check for accuracy
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]


def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['loss'] == DEFAULT_LOSS
    assert results['model_string'].startswith("mn_AFalse_d128_H128_e128_E10_rFalse_N4_n1_P64_tFalse_cpu_")
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert count_parameters(results['model'].decoder) == 1000394


def test_train_synetune():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['--st_checkpoint_dir', tmpdir])
        assert results['epoch'] == 10
        assert results['loss'] == DEFAULT_LOSS
        assert count_parameters(results['model']) == 1544650
        assert isinstance(results['model'], MotherNet)
        results = main(TESTING_DEFAULTS + ['--st_checkpoint_dir', tmpdir])
        # that we reloaded the model means we incidentally counted up to 11
        assert results['epoch'] == 11


def test_train_reload():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--save-every', '1'])
        prev_file_name = f'{results["base_path"]}/models_diff/{results["model_string"]}_epoch_2.cpkt'
        assert results['epoch'] == 2
        assert results['loss'] == pytest.approx(1.830499529838562)
        assert count_parameters(results['model']) == 1544650
        # "continue" training - will stop immediately since we already reached max epochs
        results_new = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-c', '-R'])
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
            main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-L', '2'])

        # strict loading should work if we change learning rate
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-l', '1e-3'])
        assert results['epoch'] == 2
        assert results['config']['optimizer']['learning_rate'] == 1e-3

        # non-strict loading should allow changing architecture
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-L', '2'])
        assert results['epoch'] == 2
        assert results['config']['mothernet']['predicted_hidden_layers'] == 2


def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'special_token'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert results['loss'] == pytest.approx(1.112498164176941)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "special_token"
    assert count_parameters(results['model'].decoder) == 1000266
    assert results['model'].token_embedding.shape == (1, 1, 128)


def test_train_class_tokens():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'class_tokens'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "class_tokens"
    assert count_parameters(results['model']) == 1625930
    assert count_parameters(results['model'].decoder) == 1081674
    assert results['model'].decoder.mlp[0].in_features == 1280
    assert results['loss'] == pytest.approx(2.160637378692627)


def test_train_class_average():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'class_average'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "class_average"
    assert count_parameters(results['model']) == 1625930
    assert count_parameters(results['model'].decoder) == 1081674
    assert results['model'].decoder.mlp[0].in_features == 1280
    assert results['loss'] == pytest.approx(0.791477620601654)


def test_train_simple_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'special_token_simple'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "special_token_simple"
    assert count_parameters(results['model']) == 1478602
    assert count_parameters(results['model'].decoder) == 934218
    assert results['model'].token_embedding.shape == (1, 1, 128)
    assert results['loss'] == pytest.approx(3.8096213340759277)


def test_train_average_decoder():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D', 'average'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNet)
    assert results['model'].decoder_type == "average"
    assert count_parameters(results['model']) == 1478474
    assert count_parameters(results['model'].decoder) == 934218
    assert results['loss'] == pytest.approx(1.2824537754058838)


def test_train_reduce_on_spike():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--reduce-lr-on-spike', 'True'])
    assert results['loss'] == DEFAULT_LOSS
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)


def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L', '2'])
    assert results['loss'] == pytest.approx(0.6911734938621521)
    assert count_parameters(results['model']) == 2081290
    assert isinstance(results['model'], MotherNet)


def test_train_two_decoder_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-T', '2'])
    assert isinstance(results['model'], MotherNet)
    assert count_parameters(results['model']) == 1561162
    assert results['loss'] == pytest.approx(0.6795329451560974)


def test_train_low_rank_ignored():
    # it boolean flag is not set, -W is ignored for easier hyperparameter search
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W', '16', '--low-rank-weights', 'False'])
    assert results['loss'] == DEFAULT_LOSS
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)


def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128',
                        '--experiment', 'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',
                        '--reduce-lr-on-spike', 'True', '-B', tmpdir, '-W', '16', '--low-rank-weights', 'True'])
        clf = MotherNetClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert count_parameters(results['model']) == 926474
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 2314
    # suspiciously low tolerance here
    assert results['loss'] == pytest.approx(1.1127089262008667, rel=1e-4)
    assert isinstance(results['model'], MotherNet)


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert results['loss'] == pytest.approx(1.6330838203430176, rel=1e-5)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--multiclass-type', 'steps'])
    assert results['loss'] == pytest.approx(0.7380795478820801)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only'])
    assert results['loss'] == pytest.approx(0.723449170589447)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-p-uninformative', '.9'])
    assert results['loss'] == pytest.approx(0.713407039642334)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-max-fraction-uninformative', '2'])
    assert results['loss'] == pytest.approx(0.7003865242004395)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '30', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                       'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike',
                        'True', '-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'bag_boolean'])
    assert results['loss'] == pytest.approx(0.6881492137908936)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--add-uninformative-features', 'True'])
    assert results['loss'] == pytest.approx(1.12027883529663092)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_additive_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9690634
    assert results['loss'] == pytest.approx(0.7711865901947021, rel=1e-5)


def test_train_additive_class_tokens():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--decoder-type', 'class_tokens'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert count_parameters(results['model']) == 9771914
    assert results['loss'] == pytest.approx(1.3962957859039307, rel=1e-5)


def test_train_additive_input_bin_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True', '--save-every', '2'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7029958367347717, rel=1e-5)


def test_train_additive_input_bin_embedding_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True', '--bin-embedding-rank', '8'])
    assert results['model'].encoder.embedding.shape == (64, 8)
    assert count_parameters(results['model']) == 8975018
    assert results['loss'] == pytest.approx(0.7063571810722351, rel=1e-5)


def test_train_additive_input_bin_embedding_linear():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'linear'])
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(0.7057666182518005, rel=1e-5)


def test_train_additive_factorized_output():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--save-every', '2'])
        clf = MotherNetAdditiveClassifier(device='cpu', path=get_model_path(results))
        check_predict_iris(clf)
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1649994
    assert results['loss'] == pytest.approx(2.0055530071258545, rel=1e-5)


def test_train_additive_factorized_output_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--output-rank', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64, 10)
    assert count_parameters(results['model']) == 1487514
    assert results['loss'] == pytest.approx(1.000483512878418, rel=1e-5)


def test_train_additive_factorized_in_and_out():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--input-bin-embedding', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1038090
    assert results['loss'] == pytest.approx(2.9002177715301514, rel=1e-5)


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

import tempfile

import lightning as L
import numpy as np
import pytest

from tabpfn.fit_model import main
# from tabpfn.fit_tabpfn import main as tabpfn_main
from tabpfn.models.mothernet_additive import MotherNetAdditive
from tabpfn.models.perceiver import TabPerceiver
from tabpfn.models.transformer import TabPFN
from tabpfn.models.mothernet import MotherNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']

DEFAULT_LOSS = pytest.approx(2.3810832500457764)

def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert results['loss'] == DEFAULT_LOSS
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
        assert results['loss'] == pytest.approx(2.4860925674438477)
        assert count_parameters(results['model']) == 1544650
        # "continue" training - will stop immediately since we already reached max epochs
        results_new = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-c', '-R'])
        # epoch 3 didn't actually happen, but going through the training loop raises the counter by one...
        assert results_new['epoch'] == 3
        assert results_new['loss'] == np.inf
        assert results_new['base_path'] == results['base_path']
        assert results_new['model_string'].startswith(results['model_string'])
        for k, v in results['config'].items():
            # fixme num_classes really shouldn't be a callable in config
            if k not in ['warm_start_from', 'continue_run', 'num_classes', 'num_features_used']:
                assert results_new['config'][k] == v
        # strict loading should fail if we change model arch
        with pytest.raises(RuntimeError):
            main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-L', '2'])

        # strict loading should work if we change learning rate
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-l', '1e-3'])
        assert results['epoch'] == 2
        assert results['config']['lr'] == 1e-3

        # non-strict loading should allow changing architecture
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-L', '2'])
        assert results['epoch'] == 2
        assert results['config']['predicted_hidden_layers'] == 2


def test_train_double_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert results['loss'] == pytest.approx(2.2748546600341797)
    assert count_parameters(results['model']) == 1775818
    assert isinstance(results['model'], MotherNet)


def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S', 'True'])
    assert results['loss'] == pytest.approx(2.382296562194824)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], MotherNet)
    assert results['model'].special_token
    assert count_parameters(results['model'].decoder) == 1000266
    assert results['model'].token_embedding.shape == (1, 1, 128)


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
    assert results['loss'] == pytest.approx(2.374464750289917)
    assert count_parameters(results['model']) == 2081290
    assert isinstance(results['model'], MotherNet)


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
        results = main(['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-S', 'False', '-P', '64', '-H', '128', '-d', '128',
                        '--experiment', 'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',
                        '--reduce-lr-on-spike', 'True', '-B', tmpdir, '-W', '16', '--low-rank-weights', 'True'])
    assert count_parameters(results['model']) == 926474
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 2314
    # suspiciously low tolerance here
    assert results['loss'] == pytest.approx(1.996809959411621, rel=1e-4)
    assert isinstance(results['model'], MotherNet)


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert results['loss'] == pytest.approx(2.3253633975982666, rel=1e-5)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--multiclass-type', 'steps'])
    assert results['loss'] == pytest.approx(2.34334135055542)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only'])
    assert results['loss'] == pytest.approx(2.3595831394195557)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-p-uninformative', '.9'])
    assert results['loss'] == pytest.approx(2.349252939224243)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-max-fraction-uninformative', '2'])
    assert results['loss'] == pytest.approx(2.337641716003418)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '30', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-S', 'False', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                       'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike',
                       'True', '-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'bag_boolean'])
    assert results['loss'] == pytest.approx(2.3121724128723145)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--add-uninformative-features', 'True'])
    assert results['loss'] == pytest.approx(2.329286575317383)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_heterogeneous_batches():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--heterogeneous-batches', 'True'])
    assert results['dataloader'].prior.heterogeneous_batches
    assert isinstance(results['model'], TabPFN)
    assert count_parameters(results['model']) == 579850
    assert results['loss'] == pytest.approx(2.3380088806152344)

def test_train_tabpfn_refactored():
    pytest.skip("This is not working yet")
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = tabpfn_main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert results['loss'] == 2.3345985412597656
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_additive_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive'])
    assert results['loss'] == pytest.approx(2.483247756958008, rel=1e-5)
    assert count_parameters(results['model']) == 9690634
    assert isinstance(results['model'], MotherNetAdditive)


def test_train_additive_input_bin_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(2.6669483184814453, rel=1e-5)


def test_train_additive_input_bin_embedding_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'True', '--bin-embedding-rank', '8'])
    assert results['model'].encoder.embedding.shape == (64, 8)
    assert count_parameters(results['model']) == 8975018
    assert results['loss'] == pytest.approx(2.458144187927246, rel=1e-5)

def test_train_additive_input_bin_embedding_linear():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--input-bin-embedding', 'linear'])
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert count_parameters(results['model']) == 9078730
    assert results['loss'] == pytest.approx(2.427090883255005, rel=1e-5)

def test_train_additive_factorized_output():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1649994
    assert results['loss'] == pytest.approx(4.083449363708496, rel=1e-5)


def test_train_additive_factorized_output_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--output-rank', '4'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].decoder.output_weights.shape == (4, 64, 10)
    assert count_parameters(results['model']) == 1487514
    assert results['loss'] == pytest.approx(3.8715691566467285, rel=1e-5)


def test_train_additive_factorized_in_and_out():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-m', 'additive', '--factorized-output', 'True', '--input-bin-embedding', 'True'])
    assert isinstance(results['model'], MotherNetAdditive)
    assert results['model'].encoder.embedding.shape == (64, 16)
    assert results['model'].decoder.output_weights.shape == (16, 64, 10)
    assert count_parameters(results['model']) == 1038090
    assert results['loss'] == pytest.approx(3.627715587615967, rel=1e-5)


def test_train_perceiver_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    model = results['model']
    assert isinstance(model, TabPerceiver)
    assert model.ff_dropout == 0
    assert model.no_double_embedding
    assert model.input_dim == 128
    assert len(model.layers) == 4
    assert model.latents.shape == (512, 128)
    assert model.decoder.hidden_size == 128
    assert model.decoder.emsize == 128
    assert count_parameters(model.decoder) == 1000394
    assert count_parameters(model.encoder) == 12928
    assert count_parameters(model.layers) == 664576
    assert count_parameters(model) == 1744842
    assert results['loss'] == pytest.approx(2.4952123165130615)

def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert isinstance(results['model'], TabPerceiver)
    assert count_parameters(results['model']) == 2281482
    assert results['loss'] == pytest.approx(2.0544097423553467)


def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16', '--low-rank-weights', 'True'])
    assert isinstance(results['model'], TabPerceiver)
    assert results['model'].decoder.shared_weights[0].shape == (16, 64)
    assert results['model'].decoder.mlp[2].out_features == 2314
    assert count_parameters(results['model']) == 1126666
    assert results['loss'] == pytest.approx(1.6826519966125488, rel=1e-5)
import tempfile

import lightning as L
import numpy as np
import pytest

from tabpfn.fit_model import main
from tabpfn.fit_tabpfn import main as tabpfn_main
from tabpfn.models.mothernet_additive import MotherNetAdditive
from tabpfn.models.perceiver import TabPerceiver
from tabpfn.models.transformer import TransformerModel
from tabpfn.models.transformer_make_model import TransformerModelMakeMLP


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'True', '-e', '128', '-N', '4', '-S', 'False', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike', 'True']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'True', '-e', '128', '-S', 'False', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike', 'True']


def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert results['loss'] == pytest.approx(2.4058380126953125)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_synetune():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['--st_checkpoint_dir', tmpdir])
        assert results['epoch'] == 10
        assert results['loss'] == pytest.approx(2.4058380126953125)
        assert count_parameters(results['model']) == 1544650
        assert isinstance(results['model'], TransformerModelMakeMLP)
        results = main(TESTING_DEFAULTS + ['--st_checkpoint_dir', tmpdir])
        # that we reloaded the model means we incidentally counted up to 11
        assert results['epoch'] == 11


def test_train_reload():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--save-every', '1'])
        prev_file_name = f'{results["base_path"]}/models_diff/{results["model_string"]}_epoch_2.cpkt'
        assert results['epoch'] == 2
        assert results['loss'] == 2.6038432121276855
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
            if k not in ['warm_start_from', 'continue_old_config', 'num_classes', 'num_features_used']:
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
    assert results['loss'] == 2.2983956336975098
    assert count_parameters(results['model']) == 1775818
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S', 'True'])
    assert results['loss'] == 2.2754929065704346
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_reduce_on_spike():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--reduce-lr-on-spike', 'True'])
    assert results['loss'] == pytest.approx(2.4058380126953125)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L', '2'])
    assert results['loss'] == 2.298816442489624
    assert count_parameters(results['model']) == 2081290
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_low_rank_ignored():
    # it boolean flag is not set, -W is ignored for easier hyperparameter search
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W', '16', '--low-rank-weights', 'False'])
    assert results['loss'] == pytest.approx(2.4058380126953125)
    assert count_parameters(results['model']) == 1544650
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '10', '-n', '1', '-A', 'True', '-e', '128', '-N', '4', '-S', 'False', '-P', '64', '-H', '128', '-d', '128',
                        '--experiment', 'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',
                        '--reduce-lr-on-spike', 'True', '-B', tmpdir, '-W', '16', '--low-rank-weights', 'True'])
    assert results['loss'] == pytest.approx(2.299163341522217)
    assert count_parameters(results['model']) == 926474
    assert isinstance(results['model'], TransformerModelMakeMLP)


def test_train_tabpfn():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert results['loss'] == 2.3182566165924072
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TransformerModel)


def test_train_tabpfn_refactored():
    pytest.skip("This is not working yet")
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = tabpfn_main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert results['loss'] == 2.3345985412597656
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TransformerModel)


def test_train_additive_defaults():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive'])
    assert results['loss'] == pytest.approx(3.025623321533203, rel=1e-5)
    assert count_parameters(results['model']) == 9690634
    assert isinstance(results['model'], MotherNetAdditive)


def test_train_additive_shared_embedding():
    pytest.skip("This is not working yet")
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'additive', '--shared-embedding', 'True'])
    assert results['loss'] == 2.110102653503418
    assert count_parameters(results['model']) == 9690634
    assert isinstance(results['model'], MotherNetAdditive)


def test_train_perceiver_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    assert results['loss'] == pytest.approx(2.2145862579345703)
    assert count_parameters(results['model']) == 1744842
    assert isinstance(results['model'], TabPerceiver)


def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert results['loss'] == pytest.approx(1.9268087148666382)
    assert count_parameters(results['model']) == 2281482
    assert isinstance(results['model'], TabPerceiver)


def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16', '--low-rank-weights', 'True'])
    assert results['loss'] == pytest.approx(2.252650737762451, rel=1e-5)
    assert count_parameters(results['model']) == 1126666
    assert isinstance(results['model'], TabPerceiver)

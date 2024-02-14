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
from tabpfn.utils import compare_dicts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# really tiny model for smoke tests
# one step per epoch, no adapting batchsize, CPU, Mothernet
TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                    'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                          'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--low-rank-weights', 'False']

DEFAULT_LOSS = pytest.approx(2.379756212234497)

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
        ignored_configs = ['warm_start_from', 'continue_run']
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


def test_train_double_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert results['loss'] == pytest.approx(2.2748215198516846)
    assert count_parameters(results['model']) == 1775818
    assert isinstance(results['model'], MotherNet)


def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S', 'True'])
    assert results['loss'] == pytest.approx(2.3847053050994873)
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
    assert results['loss'] == pytest.approx(2.374462366104126)
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
    assert results['loss'] == pytest.approx(1.996809959411621)
    assert isinstance(results['model'], MotherNet)


def test_train_tabpfn_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert results['loss'] == pytest.approx(2.3253633975982666)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_stepped_multiclass():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--multiclass-type', 'steps'])
    assert results['loss'] == pytest.approx(2.3433380126953125)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only'])
    assert results['loss'] == pytest.approx(2.3440020084381104)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_p_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-p-uninformative', '.9'])
    assert results['loss'] == pytest.approx(2.359046697616577)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_prior_max_uninformative():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'boolean_only', '--boolean-max-fraction-uninformative', '1'])
    assert results['loss'] == pytest.approx(2.3682920932769775)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_boolean_mixed_prior():
    L.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['-C', '-E', '30', '-n', '1', '-A', 'True', '-e', '128', '-N', '4', '-S', 'False', '-P', '64', '-H', '128', '-d', '128', '--experiment',
                       'testing_experiment', '--no-mlflow', '--train-mixed-precision', 'False', '--min-lr', '0',  '--low-rank-weights', 'False', '--reduce-lr-on-spike',
                       'True', '-B', tmpdir, '-m', 'tabpfn', '--prior-type', 'bag_boolean'])
    assert results['loss'] == pytest.approx(2.3185601234436035)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_uninformative_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--add-uninformative-features', 'True'])
    assert results['loss'] == pytest.approx(2.3292839527130127)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


def test_train_tabpfn_heterogeneous_batches():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn', '--heterogeneous-batches', 'True'])
    assert results['loss'] == pytest.approx(2.3378920555114746)
    assert count_parameters(results['model']) == 579850
    assert isinstance(results['model'], TabPFN)


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
    assert results['loss'] == pytest.approx(2.4964823722839355, rel=1e-5)
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
    model = results['model']
    assert isinstance(model, TabPerceiver)
    assert model.input_dim == 128
    assert len(model.layers) == 4
    assert model.latents.shape == (512, 512)
    assert count_parameters(model) == 1744842
    assert results['loss'] == pytest.approx(2.4954166412353516)
    
    
def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert isinstance(results['model'], TabPerceiver)
    assert count_parameters(results['model']) == 2281482
    assert results['loss'] == pytest.approx(2.054527997970581)


def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16', '--low-rank-weights', 'True'])
    assert isinstance(results['model'], TabPerceiver)
    assert count_parameters(results['model']) == 1126666
    assert results['loss'] == pytest.approx(1.6826262474060059, rel=1e-5)

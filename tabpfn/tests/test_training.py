import tempfile
import pytest
import numpy as np

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
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', '-e', '128', '-N', '4', '-P', '64', '-H', '128', '-d', '128', '--experiment', 'testing_experiment']

def test_train_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _, _, _, _ = main(TESTING_DEFAULTS + ['-B', tmpdir])
    assert loss == 2.4132816791534424
    assert count_parameters(model) == 1544650
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_reload():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, _, config, base_path, model_string, epoch = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--save-every', '1'])
        prev_file_name = f'{base_path}/models_diff/{model_string}_epoch_2.cpkt'
        assert epoch == 2
        assert loss == 2.6263158321380615
        assert count_parameters(model) == 1544650
        # "continue" training - will stop immediately since we already reached max epochs
        new_loss, new_model, _, new_config, new_base_path, new_model_string, new_epoch = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-c'])
        # epoch 3 didn't actually happen, but going through the training loop raises the counter by one...
        assert new_epoch == 3
        assert new_loss == np.inf
        assert new_base_path == base_path
        assert new_model_string == model_string
        for k, v in config.items():
            # fixme num_classes really shouldn't be a callable in config
            if k not in ['warm_start_from', 'continue_old_config', 'num_classes', 'num_features_used']:
                assert new_config[k] == v
        # strict loading should fail if we change model arch
        with pytest.raises(RuntimeError):
            new_loss, new_model, _, new_config, new_base_path, new_model_string, new_epoch = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-L', '2'])

        # strict loading should work if we change learning rate
        new_loss, new_model, _, new_config, new_base_path, new_model_string, new_epoch = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-s', '-l', '1e-3'])
        assert new_epoch == 2
        assert new_config['lr'] == 1e-3

        # non-strict loading should allow changing architecture
        new_loss, new_model, _, new_config, new_base_path, new_model_string, new_epoch = main(TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '-f', prev_file_name, '-L', '2'])
        assert new_epoch == 2
        assert new_config['predicted_hidden_layers'] == 2
    

def test_train_double_embedding():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-D'])
    assert loss == 2.2314932346343994
    assert count_parameters(model) == 1775818
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_special_token():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-S'])
    assert loss == 2.3119993209838867
    assert count_parameters(model) == 1544650
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-L', '2'])
    assert loss == 2.3295023441314697
    assert count_parameters(model) == 2081290
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-W', '16'])
    assert loss == 2.3065733909606934
    assert count_parameters(model) == 926474
    assert isinstance(model, TransformerModelMakeMLP)

def test_train_tabpfn():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'tabpfn'])
    assert loss == 2.3345985412597656
    assert count_parameters(model) == 579850
    assert isinstance(model, TransformerModel)

def test_train_perceiver_defaults():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver'])
    assert loss == 2.3633618354797363
    assert count_parameters(model) == 1744842
    assert isinstance(model, TabPerceiver)

def test_train_perceiver_two_hidden_layers():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-L', '2'])
    assert loss == 2.0139236450195312
    assert count_parameters(model) == 2281482
    assert isinstance(model, TabPerceiver)

def test_train_perceiver_low_rank():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        loss, model, *_ = main(TESTING_DEFAULTS + ['-B', tmpdir, '-m', 'perceiver', '-W', '16'])
    assert loss == 2.256040334701538
    assert count_parameters(model) == 1126666
    assert isinstance(model, TabPerceiver)
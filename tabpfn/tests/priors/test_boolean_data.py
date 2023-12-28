from tabpfn.priors.boolean_conjunctions import BooleanConjunctionSampler
import torch
import pytest

@pytest.mark.parametrize('num_features', [1, 2, 10, 100])
@pytest.mark.parametrize('seq_len', [10, 100, 1000])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_boolean_data(num_features, seq_len, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise pytest.skip("CUDA not available")
    # test call (which has padding)
    x, y, sample_params = BooleanConjunctionSampler()(seq_len=seq_len, num_features=num_features, device=device)
    assert x.shape == (seq_len, 1, num_features)
    assert y.shape == (seq_len, 1, 1)
    assert sample_params['num_features'] == num_features
    assert sample_params['num_features_active'] <= sample_params['num_features']
    assert sample_params['num_features_important'] <= sample_params['num_features_active']
    assert sample_params['features_in_terms'].sum() <=sample_params['num_features_important']
    assert (x[:, sample_params['num_features_active']:] == 0).all()
from tabpfn.priors.boolean_conjunctions import sample_boolean_data
import torch
import pytest

@pytest.mark.parametrize('num_features', [1, 2, 10, 100])
@pytest.mark.parametrize('seq_len', [10, 100, 1000])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_boolean_data(num_features, seq_len, device):
    x, y = sample_boolean_data({}, num_features=num_features, seq_len=seq_len, device=device)
    assert x.shape == (seq_len, num_features)
    assert y.shape == (seq_len,)
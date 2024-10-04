from ticl.priors.boolean_conjunctions import BooleanConjunctionPrior
import torch
import pytest
import lightning as L


@pytest.mark.parametrize('num_features', [1, 2, 10, 100])
@pytest.mark.parametrize('n_samples', [10, 100, 1000])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('max_fraction_uninformative', [0, 0.5, 1])
@pytest.mark.parametrize('p_uninformative', [0, 0.5, 1])
def test_boolean_data(num_features, n_samples, device, max_fraction_uninformative, p_uninformative):
    L.seed_everything(42)
    if device == "cuda" and not torch.cuda.is_available():
        raise pytest.skip("CUDA not available")
    # test call (which has padding)
    hyperparameters = {'max_fraction_uninformative': max_fraction_uninformative, 'p_uninformative': p_uninformative}
    x, y, sample_params = BooleanConjunctionPrior(hyperparameters=hyperparameters, debug=True)(n_samples=n_samples, num_features=num_features, device=device)
    assert x.shape == (n_samples, 1, num_features)
    assert y.shape == (n_samples, 1, 1)
    assert sample_params['num_features'] == num_features
    assert sample_params['num_features_active'] <= sample_params['num_features']
    assert sample_params['num_features_important'] <= sample_params['num_features_active']
    assert len(sample_params['features_in_terms']) <= sample_params['num_features_important']
    assert sample_params['num_features_important'] >= max_fraction_uninformative * \
        (sample_params['num_features_active'] - sample_params['num_features_important'])
    if p_uninformative == 0:
        assert sample_params['num_features_important'] == sample_params['num_features_active']
    if p_uninformative == 1 and max_fraction_uninformative > 0 and sample_params['num_features'] > 1:
        assert sample_params['num_features_important'] < sample_params['num_features_active']
    assert (x[:, sample_params['num_features_active']:] == 0).all()
    assert .1 <= y.float().mean() <= .8

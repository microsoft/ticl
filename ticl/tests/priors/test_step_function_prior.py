import lightning as L
import pytest
import torch

from ticl.priors.step_function_prior import StepFunctionPrior


@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_step_function_prior(num_features, batch_size, n_samples):
    L.seed_everything(42)
    prior = StepFunctionPrior({'max_steps': 1, 'sampling': 'uniform'})
    x, y, step_function, step, mask = prior._get_batch(batch_size=batch_size, n_samples=n_samples,
                                                       num_features=num_features, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)

    # Iterate over each feature
    for batch_i in range(batch_size):
        for feature_i in range(step_function.shape[2]):
            if mask[batch_i, feature_i]:
                # Check that the step function is zero for all samples
                assert torch.all(step_function[batch_i, :, feature_i] == 0)
            else:
                x_i, step_i = zip(*sorted(zip(x[:, batch_i, feature_i], step_function[batch_i, :, feature_i])))
                step_i = torch.Tensor(step_i).long()
                # Count the number of steps in the feature
                steps = sum(step_i[1:] != step_i[:-1])
                # Check that there is only a single step
                assert steps == 1, f"Feature {feature_i} has {steps} steps"

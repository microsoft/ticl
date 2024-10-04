import torch

from ticl.distributions import parse_distributions, sample_distributions
from ticl.utils import default_device


class StepFunctionPrior:
    def __init__(self, config=None):
        self.config = parse_distributions(config or {})

    def _get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None,
                  single_eval_pos=None):
        # small wrapper to sample from the prior but also return the individual step functions.
        with torch.no_grad():
            hypers = sample_distributions(self.config)
            if hypers['sampling'] == 'uniform':
                x = torch.rand(batch_size, n_samples, num_features, device=device)
            else:
                x = torch.randn(batch_size, n_samples, num_features, device=device)

            # Per element in batch create a random step function per feature and add them across samples.
            step = (x.max(dim=1).values - x.min(dim=1).values) * 0.8 * torch.rand((batch_size, num_features), device=device) + x.min(dim=1).values

            # Randomly sample up two features
            feature_indices = torch.randint(0, num_features, (batch_size, 2), device=device)

            # Create a mask that is True for the indices that are not in feature_indices
            mask = torch.ones_like(step, dtype=torch.bool, device=device)
            mask.scatter_(1, feature_indices, False)

            # Use the mask to set the corresponding features in step to -inf
            step[mask] = -torch.inf

            # Compute the step function for the entire tensor at once
            step_function = (x < step.unsqueeze(1)).int()
            y = (step_function.float().sum(dim=-1) > 0).float()

            x = x.permute(1, 0, 2)  # (n_samples, batch_size, num_features)
            y = y.permute(1, 0)  # (n_samples, batch_size)
            return x, y, step_function, step, mask

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None,
                  single_eval_pos=None):
        x, y, _, _, _ = self._get_batch(batch_size, n_samples, num_features, device, num_outputs, epoch,)
        return x, y, y, {}
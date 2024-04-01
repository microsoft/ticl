import torch

from mothernet.distributions import parse_distributions, sample_distributions
from mothernet.utils import default_device


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
            up_or_down = torch.randint(0, 2, size=(batch_size, num_features), device=device).unsqueeze(dim=1)
            step = (x.max(dim=1).values - x.min(dim=1).values) * 0.8 * torch.rand((batch_size, num_features)) + x.min(dim=1).values

            # Compute the step function for the entire tensor at once
            step_function = ((x < step.unsqueeze(1)).int() + up_or_down) % 2
            y = (step_function.float().sum(dim=-1) > 0).float()

            x = x.permute(1, 0, 2)  # (n_samples, batch_size, num_features)
            y = y.permute(1, 0)  # (n_samples, batch_size)
            return x, y, step_function, step

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None,
                  single_eval_pos=None):
        x, y, _, _ = self._get_batch(batch_size, n_samples, num_features, device, num_outputs, epoch,)
        return x, y, y
import torch

from tabpfn.priors.utils import eval_simple_dist

class BagPrior:
    def __init__(self, base_priors, prior_weights, verbose=False):
        self.base_priors = base_priors
        # let's make sure we get consistent sorting of the base priors by name
        self.prior_names = sorted(base_priors.keys())
        self.prior_weights = prior_weights
        self.verbose = verbose


    def get_batch(self, *, batch_size, n_samples, num_features, device, hyperparameters, batch_size_per_prior_sample=None, epoch=None, single_eval_pos=None):
        batch_size_per_prior_sample = batch_size_per_prior_sample or (min(64, batch_size))
        num_models = batch_size // batch_size_per_prior_sample
        assert num_models * \
            batch_size_per_prior_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_prior_sample ({batch_size_per_prior_sample})'

        args = {'device': device, 'n_samples': n_samples, 'num_features': num_features,
                'batch_size': batch_size_per_prior_sample, 'epoch': epoch, 'single_eval_pos': single_eval_pos}

        weights = torch.tensor([self.prior_weights[prior_name] for prior_name in self.prior_names], dtype=torch.float)
        weights = weights / torch.sum(weights)
        batch_assignments = torch.multinomial(weights, num_models, replacement=True).numpy()
        if self.verbose or 'verbose' in hyperparameters and hyperparameters['verbose']:
            print('PRIOR_BAG:', weights, batch_assignments, torch.softmax(weights, 0))

        sample = [self.base_priors[self.prior_names[int(prior_idx)]].get_batch(hyperparameters=hyperparameters, **args) for prior_idx in batch_assignments]
        x, y, y_ = zip(*sample)
        x, y, y_ = (torch.cat(x, 1).detach(), torch.cat(y, 1).detach(), torch.cat(y_, 1).detach())
        return x, y, y_
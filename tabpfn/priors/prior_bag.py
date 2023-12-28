import torch

from tabpfn.utils import default_device
from tabpfn.priors.utils import eval_simple_dist

class BagPrior:
    def __init__(self, base_priors, prior_exp_weights, verbose=False):
        self.base_priors = base_priors
        # let's make sure we get consistent sorting of the base priors by name
        self.prior_names = sorted(base_priors.keys())
        self.prior_exp_weights = prior_exp_weights
        self.verbose = verbose


    def get_batch(self, *, batch_size, n_samples, num_features, device, hyperparameters, batch_size_per_prior_sample=None, epoch=None, single_eval_pos=None):
        batch_size_per_prior_sample = batch_size_per_prior_sample or (min(64, batch_size))
        num_models = batch_size // batch_size_per_prior_sample
        assert num_models * \
            batch_size_per_prior_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_prior_sample ({batch_size_per_prior_sample})'

        args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'batch_size': batch_size_per_prior_sample, 'epoch': epoch, 'single_eval_pos': single_eval_pos}

        prior_bag_priors_p = [eval_simple_dist(self.prior_exp_weights.get(prior_name, 1)) for prior_name in self.prior_names]

        weights = torch.tensor(prior_bag_priors_p, dtype=torch.float)
        batch_assignments = torch.multinomial(torch.softmax(weights, 0), num_models, replacement=True).numpy()

        if self.verbose or 'verbose' in hyperparameters and hyperparameters['verbose']:
            print('PRIOR_BAG:', weights, batch_assignments)

        sample = [self.base_priors[self.prior_names[int(prior_idx)]].get_batch(hyperparameters=hyperparameters, **args) for prior_idx in batch_assignments]
        x, y, y_ = zip(*sample)
        x, y, y_ = (torch.cat(x, 1).detach(), torch.cat(y, 1).detach(), torch.cat(y_, 1).detach())
        return x, y, y_
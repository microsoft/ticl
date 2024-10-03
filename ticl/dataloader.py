import numpy as np
from torch.utils.data import DataLoader

import ticl.priors as priors
from ticl.priors import ClassificationAdapterPrior, BagPrior, BooleanConjunctionPrior, StepFunctionPrior

import wandb
from tqdm import tqdm

class PriorDataLoader(DataLoader):
    def __init__(
        self, 
        prior, 
        num_steps, 
        batch_size, 
        min_eval_pos, 
        n_samples, 
        device, 
        num_features,
        model = None,
        random_n_samples = False,
        n_test_samples = False,
    ):

        if (random_n_samples and not n_test_samples) or (not random_n_samples and n_test_samples):
            raise ValueError("random_n_samples and test_samples must be set together.")
        
        self.prior = prior
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos

        self.random_n_samples = random_n_samples
        if random_n_samples:
            self.n_samples = np.random.randint(min_eval_pos, random_n_samples)
            self.n_test_samples = n_test_samples
        else:
            self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.epoch_count = 0
        self.model = model

    def gbm(self, epoch=None, tqdm_bar=None, single_eval_pos = None):
        # Actually can only sample up to n_samples-1
        if self.random_n_samples:
            single_eval_pos = self.n_samples - self.n_test_samples
            
        # comment this for debug
        if single_eval_pos is None:
            single_eval_pos = np.random.randint(self.min_eval_pos, self.n_samples)
        # single_eval_pos = 31496
        
        # change the description of the progress bar
        if tqdm_bar is not None:
            tqdm_bar.set_description(f'| train sample number: {single_eval_pos} | test sample number: {self.n_samples - single_eval_pos}')
        if wandb.run is not None:
            wandb.log({'train_train_sample_number': single_eval_pos, 'train_test_sample_number': self.n_samples - single_eval_pos})
        
        batch = self.prior.get_batch(
            batch_size=self.batch_size, 
            n_samples=self.n_samples, 
            num_features=self.num_features, 
            device=self.device,
            epoch=epoch,
            single_eval_pos=single_eval_pos,
        )
        # we return sampled hyperparameters from get_batch for testing but we don't want to use them as style.
        x, y, target_y, info = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
        return (info, x, y), target_y, single_eval_pos

    def __len__(self):
        return self.num_steps

    def get_test_batch(self):  # does not increase epoch_count
        return self.gbm(epoch=self.epoch_count)
    
    def iter_safe_gbm(self):
        tqdm_bar = tqdm(range(self.num_steps))
        for _ in tqdm_bar:
            try:
                yield self.gbm(epoch=self.epoch_count - 1, tqdm_bar = tqdm_bar)
            except AssertionError:
                continue

    def __iter__(self):
        self.epoch_count += 1
        return iter(self.iter_safe_gbm())


def get_dataloader(prior_config, dataloader_config, device, model = None):

    prior_type = prior_config['prior_type']
    gp_flexible = ClassificationAdapterPrior(priors.GPPrior(prior_config['gp']), num_features=prior_config['num_features'], **prior_config['classification'])
    mlp_flexible = ClassificationAdapterPrior(priors.MLPPrior(prior_config['mlp']), num_features=prior_config['num_features'], **prior_config['classification'])

    if prior_type == 'prior_bag':
        # Prior bag combines priors
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                         prior_weights={'mlp': 0.961, 'gp': 0.039})
    elif prior_type == "step_function":
        prior = priors.StepFunctionPrior(prior_config['step_function'])
    elif prior_type == "boolean_only":
        prior = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
    elif prior_type == "bag_boolean":
        boolean = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'boolean': boolean},
                         prior_weights={'mlp': 0.9, 'gp': 0.02, 'boolean': 0.08})
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    return PriorDataLoader(
        prior=prior, 
        n_samples=prior_config['n_samples'],
        device=device, 
        num_features=prior_config['num_features'], 
        model = model,
        **dataloader_config
    )

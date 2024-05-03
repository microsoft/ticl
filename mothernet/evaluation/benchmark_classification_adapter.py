import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from mothernet import priors
from mothernet.model_configs import get_prior_config
from mothernet.priors import ClassificationAdapterPrior

prior_config = get_prior_config()
prior_config['prior']['classification']['num_classes'] = 10
prior_config['prior']['classification']['categorical_feature_p'] = 1.0

results = defaultdict(list)
for device in ['cpu', 'cuda']:
    for per_dataset_categorical in [True, False]:
        prior_config['prior']['classification']['per_dataset_categorical'] = per_dataset_categorical
        for batch_size in tqdm(np.linspace(2, 128, 10)):
            batch_size = int(batch_size)
            for _ in range(10):
                start = time.time()
                ClassificationAdapterPrior(
                    priors.MLPPrior(prior_config['prior']['mlp']),
                    num_features=prior_config['prior']['num_features'], device=device,
                    **prior_config['prior']['classification']
                ).get_batch(batch_size=batch_size, n_samples=500, num_features=64,
                            device=device, epoch=None, single_eval_pos=None)
                end = time.time()
                results['per_dataset_categorical'].append(per_dataset_categorical)
                results['batch_size'].append(batch_size)
                results['time'].append(end - start)

    with open(f'benchmark_classification_adapter_{device}.json', 'w') as f:
        json.dump(results, f)

for device in ['cpu', 'cuda']:
    with open(f'benchmark_classification_adapter_{device}.json', 'r') as f:
        results = json.load(f)
    sns.lineplot(x='batch_size', y='time', hue='per_dataset_categorical', data=results)
    plt.title('Device ' + device)
    plt.show()

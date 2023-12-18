import logging

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import choice, lograndint, loguniform, randint, uniform
from syne_tune.optimizer.baselines import ASHA, MOBSTER

root = logging.getLogger()
root.setLevel(logging.DEBUG)

# hyperparameter search space to consider
config_space = {
    'learning-rate': loguniform(1e-5, 1e-1),
    'n-layers': randint(1, 10),
    'hidden-size': lograndint(4, 2048),
    'dropout-rate': uniform(0, 1),
    'weight_decay': loguniform(1e-7, 1e-1),
    'onehot': choice([True, False]),
    'nonlinearity': choice(['relu', 'tanh']),
    'epochs': 10000,
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_parity.py'),
    scheduler=MOBSTER(
        config_space,
        metric='accuracy',
        resource_attr='epoch',
        max_resource_attr="epochs",
        search_options={'debug_log': False},
        mode='max',
    ),
    results_update_interval=5,
    stop_criterion=StoppingCriterion(max_wallclock_time=30 * 60),
    n_workers=16,  # how many trials are evaluated in parallel
    tuner_name="satimage-first-try"
)
tuner.run()

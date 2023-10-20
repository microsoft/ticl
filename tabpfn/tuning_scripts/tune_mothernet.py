import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, lograndint, choice
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.DEBUG)

# hyperparameter search space to consider
config_space = {
    'em-size': lograndint(32, 1024),
    'learning-rate': loguniform(1e-7, 1e-1),
    'epochs': 4000,
    'num-layers': randint(2, 24),
    'batch-size': lograndint(1, 256),
    'weight_decay': loguniform(1e-7, 1e-1),
    'adaptive-batch-size': choice([True, False]),
    'weight-decay': loguniform(1e-9, 1e-1),
    'num-predicted-hidden-layers': randint(1, 6),
    'weight-embedding-rank': choice([32, 64, 512, None]),
    'predicted-hidden-layer-size': lograndint(32, 1024),
    'learning-rate-schedule': choice(['cosine', 'constant', 'linear', 'exponential']),
    'adam-beta1': uniform(0.8, 0.999),
    'lr-decay': uniform(0.5, 1),
    'reduce-lr-on-spike': choice([True, False]),
    'save-every': 1,
    'spike-tolerance': randint(1, 10),
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='fit_model.py'),
    scheduler=MOBSTER(
        config_space,
        metric='accuracy',
        resource_attr='epoch',
        max_resource_attr="epochs",
        search_options={'debug_log': False},
        mode='min',
        type="promotion",
        grace_period=5,
    ),
    results_update_interval=5,
    #stop_criterion=StoppingCriterion(max_wallclock_time=60 *60),
    stop_criterion=StoppingCriterion(max_num_trials_started=2000),
    n_workers=4,  # how many trials are evaluated in parallel
    tuner_name="mothernet-first-try"
)
tuner.run()
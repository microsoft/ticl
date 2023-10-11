import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, lograndint, choice
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.DEBUG)

# hyperparameter search space to consider
config_space = {
    'learning-rate': loguniform(1e-5, 1e-1),
    'n-layers': randint(1, 10),
    'hidden-size': lograndint(4, 2048),
    'dropout-rate': uniform(0, 1),
    'weight_decay': loguniform(1e-7, 1e-1),
    'one-hot': choice([True, False]),
    'epochs': 10000,
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_teaching_assistant.py'),
    scheduler=HyperTune(
        config_space,
        metric='accuracy',
        resource_attr='epoch',
        max_resource_attr="epochs",
        search_options={'debug_log': False},
        mode='max',
    ),
    results_update_interval=5,
    stop_criterion=StoppingCriterion(max_wallclock_time=10 *60),
    n_workers=16,  # how many trials are evaluated in parallel
    tuner_name="teaching-assistant-id-only-hyper-tune"
)
tuner.run()
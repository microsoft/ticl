import logging

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import MOBSTER


root = logging.getLogger()
root.setLevel(logging.INFO)

tuner_name = "test-timing-tune-time-7"

# hyperparameter search space to consider
config_space = {
    'parameter': randint(0, 100),
    'epochs': 1000,
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_test_timing.py'),
    scheduler=MOBSTER(
        config_space,
        metric='loss',
        resource_attr='time',
        max_resource_attr="time",
        search_options={'debug_log': False},
        mode='min',
        type="promotion",
        grace_period=2,
    ),
    max_failures=1000,
    results_update_interval=60,
    print_update_interval=120,
    # stop_criterion=StoppingCriterion(max_wallclock_time=60 *60 * 60),
    stop_criterion=StoppingCriterion(max_num_trials_started=5000),
    n_workers=4,  # how many trials are evaluated in parallel
    tuner_name=tuner_name,
    trial_backend_path=f"/synetune_checkpoints/{tuner_name}/"
)
tuner.run()

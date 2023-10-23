import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, lograndint, choice
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.INFO)

# hyperparameter search space to consider
config_space = {
    'em-size': choice([128, 256, 512, 1024]),
    'learning-rate': loguniform(1e-7, 1e-1),
    'epochs': 4000,
    'num-layers': randint(2, 24),
    #'batch-size': lograndint(1, 256),
    'batch-size': choice([2, 4, 8, 16, 32]),
    'adaptive-batch-size': choice([True, False]),
    'weight-decay': loguniform(1e-9, 1e-1),
    'num-predicted-hidden-layers': randint(1, 6),
    'weight-embedding-rank': choice(['32', '64', '512']),
    'predicted-hidden-layer-size': lograndint(32, 1024),
    'learning-rate-schedule': choice(['cosine', 'constant', 'exponential']),
    'adam-beta1': uniform(0.8, 0.999),
    'lr-decay': uniform(0.5, 1),
    'reduce-lr-on-spike': choice([True, False]),
    'save-every': 1,
    'spike-tolerance': randint(1, 10),
    'experiment': 'synetune-mothernet-try-4',
    'warmup-epochs': randint(0, 30),
}
tuner_name = "mothernet-try-4"
tuner = Tuner(
    trial_backend=LocalBackend(entry_point='../fit_model.py'),
        scheduler=MOBSTER(
        config_space,
        metric='loss',
        resource_attr='epoch',
        max_resource_attr="epochs",
        search_options={'debug_log': False},
        mode='min',
        type="promotion",
        grace_period=5,
    ),
    max_failures=1000,
    results_update_interval=60,
    print_update_interval=120,
    #stop_criterion=StoppingCriterion(max_wallclock_time=60 *60),
    stop_criterion=StoppingCriterion(max_num_trials_started=36),
    n_workers=4,  # how many trials are evaluated in parallel
    tuner_name=tuner_name,
    trial_backend_path=f"/datadrive/synetune/{tuner_name}/"
)
tuner.run()
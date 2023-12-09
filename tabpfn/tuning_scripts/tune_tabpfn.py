import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, lograndint, choice, logfinrange
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.INFO)

tuner_name = "tabpfn-timed-minlr"


# hyperparameter search space to consider
config_space = {
    'em-size': logfinrange(lower=128, upper=1024, size=4, cast_int=True),
    'learning-rate': loguniform(1e-7, 1e-2),
    'epochs': 4000,
    'num-layers': randint(2, 24),
    'batch-size': logfinrange(lower=2, upper=32, size=5, cast_int=True),
    'adaptive-batch-size': choice([True, False]),
    'weight-decay': loguniform(1e-9, 1e-1),
    'learning-rate-schedule': choice(['cosine', 'constant', 'exponential']),
    'adam-beta1': uniform(0.8, 0.999),
    'lr-decay': uniform(0.5, 1),
    'reduce-lr-on-spike': choice([False]),
    'min-lr': loguniform(1e-8, 1e-2),
    'save-every': 1,
    #'spike-tolerance': randint(1, 10),
    'experiment': f'synetune-{tuner_name}',
    'warmup-epochs': randint(0, 30),
    #'num-steps': 128,
    'model-maker': 'tabpfn',
}

early_checkpoint_removal_kwargs = {"max_num_checkpoints": 80}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='../fit_model.py'),
        scheduler=MOBSTER(
        config_space,
        metric='loss',
        resource_attr='wallclock_time',
        max_resource_attr="wallclock_time",
        search_options={'debug_log': False},
        mode='min',
        type="promotion",
        grace_period=50,  # each tick is 5 minutes
        early_checkpoint_removal_kwargs=early_checkpoint_removal_kwargs,
    ),
    max_failures=1000,
    results_update_interval=60,
    print_update_interval=120,
    #stop_criterion=StoppingCriterion(max_wallclock_time=60 *60),
    stop_criterion=StoppingCriterion(max_num_trials_started=5000),
    n_workers=4,  # how many trials are evaluated in parallel
    tuner_name=tuner_name,
    trial_backend_path=f"/synetune_checkpoints/{tuner_name}/"
)
tuner.run()
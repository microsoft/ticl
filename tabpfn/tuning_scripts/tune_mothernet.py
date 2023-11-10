import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, choice, logfinrange
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.INFO)

tuner_name = "mothernet-big-searchspace-timed3"


# hyperparameter search space to consider
config_space = {
    'em-size': logfinrange(lower=128, upper=1024, size=4, cast_int=True),
    'learning-rate': loguniform(1e-7, 1e-1),
    'epochs': 4000,
    'num-layers': randint(2, 24),
    'batch-size': logfinrange(lower=2, upper=32, size=5, cast_int=True),
    'adaptive-batch-size': choice([True, False]),
    'decoder-two-hidden-layers': choice([True, False]),
    'special-token': choice([True, False]),
    'weight-decay': loguniform(1e-9, 1e-1),
    'num-predicted-hidden-layers': randint(1, 6),
    'low-rank-weights': choice([True, False]),
    'weight-embedding-rank': logfinrange(lower=16, upper=512, size=6, cast_int=True),
    'predicted-hidden-layer-size': logfinrange(lower=32, upper=1024, size=6, cast_int=True),
    'learning-rate-schedule': choice(['exponential']),
    'adam-beta1': uniform(0.8, 0.999),
    'lr-decay': uniform(0.90, 0.9999),
    'reduce-lr-on-spike': choice([True, False]),
    'save-every': 1,
    'spike-tolerance': randint(1, 10),
    'experiment': f'synetune-{tuner_name}',
    'warmup-epochs': randint(0, 30),
    'decoder-em-size': logfinrange(lower=128, upper=4096, size=6, cast_int=True),
    'decoder-hidden-size': logfinrange(lower=128, upper=4096, size=6, cast_int=True),
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
        grace_period=10,
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
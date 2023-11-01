import logging

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, uniform, lograndint, choice, logfinrange
from syne_tune.optimizer.baselines import ASHA, MOBSTER, HyperTune
root = logging.getLogger()
root.setLevel(logging.INFO)

tuner_name = "perceiver_first_try"


# hyperparameter search space to consider
config_space = {
    'em-size': logfinrange(lower=128, upper=1024, size=4, cast_int=True),
    'learning-rate': loguniform(1e-7, 1e-1),
    'epochs': 4000,
    'num-layers': randint(2, 24),
    'gpu-id': 2,
    'batch-size': logfinrange(lower=2, upper=32, size=5, cast_int=True),
    'adaptive-batch-size': choice([True, False]),
    'weight-decay': loguniform(1e-9, 1e-1),
    'num-predicted-hidden-layers': randint(1, 6),
    'num-latents': logfinrange(lower=32, upper=1024, size=6, cast_int=True),
    'weight-embedding-rank': logfinrange(lower=32, upper=512, size=5, cast_int=True),
    'predicted-hidden-layer-size': logfinrange(lower=32, upper=1024, size=6, cast_int=True),
    'learning-rate-schedule': choice(['cosine', 'constant', 'exponential']),
    'adam-beta1': uniform(0.8, 0.999),
    'lr-decay': uniform(0.90, 0.9999),
    'reduce-lr-on-spike': choice([True]),
    'save-every': 1,
    'model-maker': 'perceiver',
    'spike-tolerance': randint(1, 10),
    'experiment': f'synetune-{tuner_name}',
    'warmup-epochs': randint(0, 30),
}
early_checkpoint_removal_kwargs = {"max_num_checkpoints": 80}


tuner = Tuner(
    trial_backend=LocalBackend(entry_point='../fit_model.py'),
        scheduler=MOBSTER(
        config_space,
        metric='loss',
        resource_attr='epoch',
        max_resource_attr="stop_after_epochs",
        search_options={'debug_log': False},
        mode='min',
        type="promotion",
        grace_period=5,
        early_checkpoint_removal_kwargs=early_checkpoint_removal_kwargs,

    ),
    max_failures=1000,
    results_update_interval=60,
    print_update_interval=120,
    #stop_criterion=StoppingCriterion(max_wallclock_time=60 *60),
    stop_criterion=StoppingCriterion(max_num_trials_started=5000),
    n_workers=1,  # how many trials are evaluated in parallel
    tuner_name=tuner_name,
    trial_backend_path=f"/synetune_checkpoints/{tuner_name}/"
)
tuner.run()
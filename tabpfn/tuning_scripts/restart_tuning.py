from syne_tune.experiments import load_experiment
import sys
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)

tuner = load_experiment(sys.argv[1], load_tuner=True).tuner
tuner.trial_backend.delete_checkpoints = True
tuner.scheduler.early_checkpoint_removal_kwargs = {"max_num_checkpoints": 80}
tuner.scheduler._initialize_early_checkpoint_removal({"max_num_checkpoints": 80})
tuner.run()
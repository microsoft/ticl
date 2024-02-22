import logging
import pdb
import sys

from syne_tune.experiments import load_experiment

root = logging.getLogger()
root.setLevel(logging.DEBUG)

tuner = load_experiment(sys.argv[1], load_tuner=True).tuner
pdb.set_trace()
tuner.max_failures = 10000
tuner.run()

import logging
import time
import os
import numpy as np

from syne_tune import Reporter
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--parameter', type=int)
    parser.add_argument('--st_checkpoint_dir', type=str, default=None)
    parser.add_argument('--stop-after-epochs', type=int)

    args = parser.parse_args()
    report = Reporter()

    for i in range(args.stop_after_epochs):
        time.sleep(args.parameter + 1)
        report(epoch=i, loss=np.exp(-1/1000 *  (np.sqrt(args.parameter + 1) * i))) 
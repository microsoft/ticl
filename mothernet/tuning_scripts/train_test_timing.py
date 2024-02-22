import os
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from syne_tune import Reporter

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--parameter', type=int)
    parser.add_argument('--st_checkpoint_dir', type=str, default=None)
    # parser.add_argument('--stop-after-epochs', type=int)
    parser.add_argument('--epochs', type=int)  # ignored

    args = parser.parse_args()
    report = Reporter()
    start = 1
    extra_time = 0
    checkpoint_dir = args.st_checkpoint_dir
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.iter"
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)
                start = state['epoch'] + 1
                extra_time = state['time']
    tick = time.time()

    for i in range(start, args.epochs):
        time.sleep(0.1 * (args.parameter + 1))
        current_time = time.time() - tick + extra_time
        report(epoch=i, loss=np.exp(-1/1000 * (np.sqrt(args.parameter + 1) * i)), time=max(1, int(current_time)))
        if checkpoint_dir is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({'epoch': i, 'time': current_time}, f)

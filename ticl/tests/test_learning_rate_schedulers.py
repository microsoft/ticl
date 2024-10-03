import tempfile

import lightning as L
import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ticl.fit_model import main
# from tabpfn.fit_tabpfn import main as tabpfn_main
from ticl.utils import ExponentialLR


@pytest.mark.parametrize('learning_rate_schedule', ['cosine', 'exponential', 'constant'])
@pytest.mark.parametrize('min_lr', [1e-10, 1e-5, 1e-3])
@pytest.mark.parametrize('base_lr', [0.01, 1e-5])
def test_min_lr(learning_rate_schedule, min_lr, base_lr):
    if base_lr < min_lr and learning_rate_schedule == 'cosine':
        # we can't get to min_lr if base_lr is already lower
        pytest.skip("Cosine learning rate is a bit weird in this case")
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    warmup_epochs = 10
    epochs = 1000
    lr_decay = 0.9
    optimizer = torch.optim.SGD(model, base_lr)

    if learning_rate_schedule == 'cosine':
        base_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    elif learning_rate_schedule == 'exponential':
        base_scheduler = ExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)
    elif learning_rate_schedule == 'constant':
        base_scheduler = ExponentialLR(optimizer, gamma=1, min_lr=min_lr)
    else:
        raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
    # add linear warmup to scheduler
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs),
                                         base_scheduler], milestones=[warmup_epochs])

    import warnings
    warnings.simplefilter("ignore", UserWarning)
    lrs = []
    for i in range(epochs):
        optimizer.step()
        scheduler.step()
        lrs.append([group['lr'] for group in optimizer.param_groups][0])

    lrs = np.array(lrs)[warmup_epochs - 1:]
    max_lr = max(min_lr, base_lr)
    assert lrs.min() >= min_lr
    assert lrs.max() < max_lr + 1e-10
    assert lrs[0] == pytest.approx(max_lr, rel=1e-5)
    if learning_rate_schedule != 'constant':
        assert lrs.min() == pytest.approx(min_lr)


TESTING_DEFAULTS = ['tabpfn', '--extra-fast-test', '-C', '-E', '20', '-U', '3', '-n', '1', '-A', 'False', '-e', '8', '-N', '2',
                    '--experiment', 'testing_experiment',  '--train-mixed-precision', 'False', '--save-every', '1000', '--learning-rate', '0.003']


def test_train_defaults():
    L.seed_everything(42)
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     results = main(TESTING_DEFAULTS + ['-B', tmpdir])
    # assert results['model'].learning_rates[-1] == pytest.approx(2.5550265339645747e-05)
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--min-lr', '1e-4'])
    # assert results['model'].learning_rates[-1] >= 0.00012 # cosine doesn't go all the way down to min_lr depending on number of epochs
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--learning-rate-schedule', 'exponential', '--lr-decay', '0.8'])
    assert results['model'].learning_rates[-1] == pytest.approx(8.444249301319689e-05)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(TESTING_DEFAULTS + ['-B', tmpdir, '--min-lr', '1e-4', '--learning-rate-schedule', 'exponential', '--lr-decay', '0.8'])
    assert results['model'].learning_rates[-1] == pytest.approx(1e-4)

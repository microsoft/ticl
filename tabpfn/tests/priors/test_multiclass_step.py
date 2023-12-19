from tabpfn.priors.flexible_categorical import MulticlassSteps
import torch
import pytest

@pytest.mark.parametrize('max_classes', [2, 3, 4])
@pytest.mark.parametrize('max_steps', [1, 2, 5, 10])
def test_multiclass_step(max_classes, max_steps):
    # samples x batch size
    batchsize = 8
    x = torch.rand((1152, batchsize))

    steps = MulticlassSteps(max_classes, max_steps=max_steps)
    num_classes = steps.num_classes
    num_steps = steps.num_steps
    classes = steps.forward(x)
    assert classes.shape == (1152, batchsize)
    assert (classes.unique() == torch.range(0, num_classes - 1)).all()
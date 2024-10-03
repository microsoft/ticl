from ticl.priors.classification_adapter import MulticlassSteps
import torch
import pytest
import lightning as L


@pytest.mark.parametrize('max_classes', [2, 3, 4])
@pytest.mark.parametrize('max_steps', [1, 2, 5, 10])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_multiclass_step(max_classes, max_steps, device):
    L.seed_everything(42)
    if device == "cuda" and not torch.cuda.is_available():
        raise pytest.skip("CUDA not available")
    batchsize = 8
    # samples x batch size
    x = torch.rand((1152, batchsize), device=device)

    steps = MulticlassSteps(max_classes, max_steps=max_steps)
    num_classes = steps.num_classes
    classes = steps(x)
    assert classes.shape == (1152, batchsize)
    assert (classes.unique().cpu() == torch.arange(0, num_classes)).all()

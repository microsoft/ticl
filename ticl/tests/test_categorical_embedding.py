from ticl.priors.classification_adapter import MulticlassSteps
import torch
import pytest
import lightning as L

from ticl.models.biattention_additive_mothernet import _determine_is_categorical


def test_categorical_embedding():
    x_src_org = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                              [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]]).unsqueeze(-1)

    expected_output = torch.tensor([[[0., 1., 1., 0., 0.],
                                     [0., 1., 1., 0., 0.]]]).unsqueeze(-1)

    output = _determine_is_categorical(x_src_org, {'categorical_features': [1, 2]})
    assert torch.equal(output, expected_output), 'Output is not as expected'

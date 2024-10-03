import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_threads():
    import torch
    torch.set_num_threads(1)
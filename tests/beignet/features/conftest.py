import pytest
import torch
from beignet.features._features import Feature


@pytest.fixture
def feature() -> Feature:
    return Feature(torch.tensor([1, 2, 3]))

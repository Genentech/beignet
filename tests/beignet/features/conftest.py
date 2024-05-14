import pytest
import torch
from yeji.features._feature import Feature


@pytest.fixture
def feature() -> Feature:
    return Feature(torch.tensor([1, 2, 3]))

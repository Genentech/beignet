import beignet.polynomial
import pytest
import torch


def test__pow():
    with pytest.raises(ValueError):
        beignet.polynomial._pow(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            5,
            4,
        )

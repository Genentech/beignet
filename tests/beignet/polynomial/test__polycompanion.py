import pytest
import torch
from beignet.polynomial import polycompanion


def test_polycompanion():
    with pytest.raises(ValueError):
        polycompanion(torch.tensor([]))

    with pytest.raises(ValueError):
        polycompanion(torch.tensor([1]))

    for i in range(1, 5):
        output = polycompanion(
            torch.tensor([0.0] * i + [1.0]),
        )

        assert output.shape == (i, i)

    output = polycompanion(
        torch.tensor([1, 2]),
    )

    assert output[0, 0] == -0.5

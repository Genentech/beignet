import beignet
import pytest
import torch


def test_laguerre_polynomial_companion():
    with pytest.raises(ValueError):
        beignet.laguerre_polynomial_companion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.laguerre_polynomial_companion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = beignet.laguerre_polynomial_companion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = beignet.laguerre_polynomial_companion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == 1.5

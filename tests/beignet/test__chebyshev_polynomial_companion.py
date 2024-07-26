import beignet
import pytest
import torch


def test_chebyshev_polynomial_companion():
    with pytest.raises(ValueError):
        beignet.chebyshev_polynomial_companion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.chebyshev_polynomial_companion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = beignet.chebyshev_polynomial_companion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = beignet.chebyshev_polynomial_companion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.5

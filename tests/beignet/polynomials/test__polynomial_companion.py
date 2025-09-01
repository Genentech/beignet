import pytest
import torch

import beignet.polynomials


def test_polynomial_companion():
    with pytest.raises(ValueError):
        beignet.polynomials.polynomial_companion(torch.tensor([]))

    with pytest.raises(ValueError):
        beignet.polynomials.polynomial_companion(torch.tensor([1]))

    for i in range(1, 5):
        output = beignet.polynomials.polynomial_companion(
            torch.tensor([0.0] * i + [1.0]),
        )

        assert output.shape == (i, i)

    output = beignet.polynomials.polynomial_companion(
        torch.tensor([1, 2]),
    )

    assert output[0, 0] == -0.5

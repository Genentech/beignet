import pytest
import torch

import beignet.polynomials


def test_legendre_polynomial_companion():
    with pytest.raises(ValueError):
        beignet.polynomials.legendre_polynomial_companion(torch.tensor([]))

    with pytest.raises(ValueError):
        beignet.polynomials.legendre_polynomial_companion(torch.tensor([1]))

    for index in range(1, 5):
        output = beignet.polynomials.legendre_polynomial_companion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    assert (
        beignet.polynomials.legendre_polynomial_companion(torch.tensor([1, 2]))[0, 0]
        == -0.5
    )

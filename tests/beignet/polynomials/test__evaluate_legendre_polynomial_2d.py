import pytest
import torch

import beignet.polynomials


def test_evaluate_legendre_polynomial_2d(float64):
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.polynomials.evaluate_polynomial(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    pytest.raises(
        ValueError,
        beignet.polynomials.evaluate_legendre_polynomial_2d,
        a,
        b[:2],
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        ),
    )

    torch.testing.assert_close(
        beignet.polynomials.evaluate_legendre_polynomial_2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.0, 2.0, 2.0]),
                torch.tensor([2.0, 2.0, 2.0]),
            ),
        ),
        x * y,
    )

    output = beignet.polynomials.evaluate_legendre_polynomial_2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        ),
    )

    assert output.shape == (2, 3)

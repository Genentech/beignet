import pytest
import torch

import beignet


def test_evaluate_chebyshev_polynomial_2d(float64):
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.evaluate_polynomial(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        beignet.evaluate_chebyshev_polynomial_2d(
            a,
            b[:2],
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial_2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        ),
        x * y,
    )

    output = beignet.evaluate_chebyshev_polynomial_2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert output.shape == (2, 3)

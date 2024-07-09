import beignet
import pytest
import torch


def test_evaluate_laguerre_polynomial_2d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.evaluate_polynomial(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError):
        beignet.evaluate_laguerre_polynomial_2d(
            a,
            b[:2],
            torch.einsum(
                "i,j->ij",
                torch.tensor([9.0, -14.0, 6.0]),
                torch.tensor([9.0, -14.0, 6.0]),
            ),
        )

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial_2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([9.0, -14.0, 6.0]),
                torch.tensor([9.0, -14.0, 6.0]),
            ),
        ),
        x * y,
    )

    output = beignet.evaluate_laguerre_polynomial_2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij", torch.tensor([9.0, -14.0, 6.0]), torch.tensor([9.0, -14.0, 6.0])
        ),
    )

    assert output.shape == (2, 3)

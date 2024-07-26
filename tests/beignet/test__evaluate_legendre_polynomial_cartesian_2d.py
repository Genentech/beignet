import beignet
import torch


def test_evaluate_legendre_polynomial_cartesian_2d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.evaluate_polynomial(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        beignet.evaluate_legendre_polynomial_cartesian_2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([2.0, 2.0, 2.0]),
                torch.tensor([2.0, 2.0, 2.0]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            x,
            y,
        ),
    )

    output = beignet.evaluate_legendre_polynomial_cartesian_2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([2.0, 2.0, 2.0]),
        ),
    )

    assert output.shape == (2, 3) * 2

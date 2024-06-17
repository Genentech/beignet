import beignet
import torch


def test_evaluate_polynomial_cartesian_2d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, x3 = x

    y1, y2, y3 = beignet.evaluate_polynomial(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        beignet.evaluate_polynomial_cartesian_2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        torch.einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    output = beignet.evaluate_polynomial_cartesian_2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 2

import beignet
import torch


def test_evaluate_polynomial_cartesian_3d():
    x = torch.rand(3, 5) * 2 - 1

    y = beignet.evaluate_polynomial(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    a, b, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        beignet.evaluate_polynomial_cartesian_3d(
            a,
            b,
            x3,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        torch.einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    output = beignet.evaluate_polynomial_cartesian_3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3) * 3

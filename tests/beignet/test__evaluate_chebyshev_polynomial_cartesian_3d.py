import beignet
import torch


def test_evaluate_chebyshev_polynomial_cartesian_3d():
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.evaluate_polynomial(
        input,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial_cartesian_3d(
            a,
            b,
            c,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
                torch.tensor([2.5, 2.0, 1.5]),
            ),
        ),
        torch.einsum(
            "i,j,k->ijk",
            x,
            y,
            z,
        ),
    )

    output = beignet.evaluate_chebyshev_polynomial_cartesian_3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
            torch.tensor([2.5, 2.0, 1.5]),
        ),
    )

    assert output.shape == (2, 3) * 3

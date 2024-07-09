import beignet
import torch


def test_evaluate_physicists_hermite_polynomial_cartesian_3d():
    c1d = torch.tensor([2.5, 1.0, 0.75])
    c3d = torch.einsum(
        "i,j,k->ijk",
        c1d,
        c1d,
        c1d,
    )

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = beignet.evaluate_polynomial(x, torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial_cartesian_3d(
            a,
            b,
            x3,
            c3d,
        ),
        torch.einsum(
            "i,j,k->ijk",
            y1,
            y2,
            y3,
        ),
    )

    z = torch.ones([2, 3])

    assert (
        beignet.evaluate_physicists_hermite_polynomial_cartesian_3d(z, z, z, c3d).shape
        == (2, 3) * 3
    )

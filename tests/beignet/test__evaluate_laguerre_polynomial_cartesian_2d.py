import torch

import beignet


def test_evaluate_laguerre_polynomial_cartesian_2d(float64):
    c1d = torch.tensor([9.0, -14.0, 6.0])
    c2d = torch.einsum("i,j->ij", c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    a, b, x3 = x
    y1, y2, y3 = beignet.evaluate_polynomial(x, torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial_cartesian_2d(
            a,
            b,
            c2d,
        ),
        torch.einsum("i,j->ij", y1, y2),
    )

    z = torch.ones([2, 3])
    assert (
        beignet.evaluate_laguerre_polynomial_cartesian_2d(
            z,
            z,
            c2d,
        ).shape
        == (2, 3) * 2
    )

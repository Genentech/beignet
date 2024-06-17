import beignet
import torch


def test_evaluate_probabilists_hermite_polynomial_cartersian_3d():
    c1d = torch.tensor([4.0, 2.0, 3.0])
    c3d = torch.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = torch.rand(3, 5) * 2 - 1
    y = beignet.evaluate_polynomial(x, torch.tensor([1.0, 2.0, 3.0]))

    a, b, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        beignet.evaluate_probabilists_hermite_polynomial_cartersian_3d(a, b, x3, c3d),
        torch.einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = torch.ones([2, 3])
    res = beignet.evaluate_probabilists_hermite_polynomial_cartersian_3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3

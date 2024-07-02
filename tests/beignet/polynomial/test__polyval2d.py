import torch
from beignet.polynomial import polyval, polyval2d


def test_polyval2d():
    x = torch.rand(3, 5) * 2 - 1

    a, b, c = x

    y1, y2, y3 = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        polyval2d(
            a,
            b,
            torch.einsum(
                "i,j->ij",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
        ),
        y1 * y2,
    )

    output = polyval2d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j->ij",
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        ),
    )

    assert output.shape == (2, 3)

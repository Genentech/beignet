import torch
from beignet.polynomial import polygrid3d, polyval


def test_polygrid3d():
    x = torch.rand(3, 5) * 2 - 1

    y = polyval(
        x,
        torch.tensor([1.0, 2.0, 3.0]),
    )

    a, b, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        polygrid3d(
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

    output = polygrid3d(
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

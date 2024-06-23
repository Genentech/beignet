import beignet.polynomial
import torch.testing


def test_polygrid3d():
    x = torch.rand([3, 5]) * 2 - 1

    x1, x2, x3 = x

    y1, y2, y3 = beignet.polynomial.evaluate_power_series_1d(
        x,
        [1.0, 2.0, 3.0],
    )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_grid_3d(
            x1,
            x2,
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

    output = beignet.polynomial.evaluate_power_series_grid_3d(
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

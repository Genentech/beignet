import beignet.polynomial
import torch


def test_hermmulx():
    torch.testing.assert_close(
        beignet.polynomial._hermmulx.multiply_physicists_hermite_series_by_x(
            torch.tensor([0]),
        ),
        torch.tensor([0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial._hermmulx.multiply_physicists_hermite_series_by_x(
            torch.tensor([1]),
        ),
        torch.tensor([0, 0.5], dtype=torch.float64),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial._hermmulx.multiply_physicists_hermite_series_by_x(
                torch.tensor(
                    [0] * index + [1],
                ),
            ),
            torch.tensor(
                [0] * (index - 1) + [index, 0, 0.5],
                dtype=torch.float64,
            ),
        )

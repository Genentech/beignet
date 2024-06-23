import beignet.polynomial
import torch


def test_chebmulx():
    torch.testing.assert_close(
        beignet.polynomial._chebmulx.multiply_chebyshev_series_by_x(
            torch.tensor([0]),
        ),
        torch.tensor([0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial._chebmulx.multiply_chebyshev_series_by_x(
            torch.tensor([1]),
        ),
        torch.tensor([0, 1], dtype=torch.float64),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial._chebmulx.multiply_chebyshev_series_by_x(
                torch.tensor(
                    [0] * index + [1],
                ),
            ),
            torch.tensor(
                [0] * (index - 1) + [0.5, 0, 0.5],
                dtype=torch.float64,
            ),
        )

import beignet.polynomial
import torch


def test_laguerre_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_roots(
            torch.tensor([1], dtype=torch.float64),
        ),
        torch.tensor([], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_roots(
            torch.tensor([0, 1], dtype=torch.float64),
        ),
        torch.tensor([1], dtype=torch.float64),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_laguerre_series(
                beignet.polynomial.laguerre_series_roots(
                    beignet.polynomial.laguerre_series_from_roots(
                        torch.linspace(0, 3, index),
                    )
                ),
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_laguerre_series(
                torch.linspace(0, 3, index),
                tolerance=0.000001,
            ),
        )

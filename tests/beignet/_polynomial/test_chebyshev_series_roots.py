import beignet.polynomial
import torch


def test_chebyshev_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_roots(
            torch.tensor([1], dtype=torch.float32),
        ),
        torch.tensor([], dtype=torch.float32),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_roots(
            torch.tensor([1, 2], dtype=torch.float32),
        ),
        torch.tensor([-0.5], dtype=torch.float32),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_chebyshev_series(
                beignet.polynomial.chebyshev_series_roots(
                    beignet.polynomial.chebyshev_series_from_roots(
                        torch.linspace(-1, 1, index),
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_chebyshev_series(
                torch.linspace(-1, 1, index),
                tolerance=1e-6,
            ),
        )

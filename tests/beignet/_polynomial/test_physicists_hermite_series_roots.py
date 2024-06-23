import beignet.polynomial
import torch


def test_physicists_hermite_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_roots([1]), []
    )
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_roots([1, 1]), [-0.5]
    )
    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_physicists_hermite_series(
                beignet.polynomial.physicists_hermite_series_roots(
                    beignet.polynomial.physicists_hermite_series_from_roots(
                        torch.linspace(-1, 1, i)
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_physicists_hermite_series(
                torch.linspace(-1, 1, i),
                tolerance=1e-6,
            ),
        )

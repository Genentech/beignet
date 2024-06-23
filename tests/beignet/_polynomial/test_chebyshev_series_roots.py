import beignet.polynomial
import numpy
import torch


def test_chebyshev_series_roots():
    torch.testing.assert_close(beignet.polynomial.chebyshev_series_roots([1]), [])
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_roots([1, 2]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        torch.testing.assert_close(
            beignet.polynomial.trim_chebyshev_series(
                beignet.polynomial.chebyshev_series_roots(
                    beignet.polynomial._chebfromroots.chebfromroots(tgt)
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

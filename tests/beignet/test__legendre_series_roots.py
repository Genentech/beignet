import beignet.polynomial
import numpy
import torch


def test_legendre_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_roots([1]),
        [],
    )

    torch.testing.assert_close(beignet.polynomial.legendre_series_roots([1, 2]), [-0.5])

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_legendre_series(
                beignet.polynomial.legendre_series_roots(
                    beignet.polynomial.legendre_series_from_roots(
                        numpy.linspace(-1, 1, i),
                    )
                ),
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_legendre_series(
                numpy.linspace(-1, 1, i),
                tolerance=0.000001,
            ),
        )

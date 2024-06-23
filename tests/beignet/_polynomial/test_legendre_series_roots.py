import beignet.polynomial
import numpy


def test_legendre_series_roots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial.legendre_series_roots([1]),
        [],
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.legendre_series_roots([1, 2]), [-0.5]
    )

    for i in range(2, 5):
        numpy.testing.assert_almost_equal(
            beignet.polynomial.trim_legendre_series(
                beignet.polynomial.legendre_series_roots(
                    beignet.polynomial.legfromroots(
                        numpy.linspace(-1, 1, i),
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_legendre_series(
                numpy.linspace(-1, 1, i),
                tolerance=1e-6,
            ),
        )

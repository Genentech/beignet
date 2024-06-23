import beignet.polynomial
import beignet.polynomial._chebfromroots
import beignet.polynomial._chebyshev_series_roots
import beignet.polynomial._trim_chebyshev_series
import numpy


def test_chebroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebroots.chebyshev_series_roots([1]), []
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebroots.chebyshev_series_roots([1, 2]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.trim_chebyshev_series(
                beignet.polynomial._chebroots.chebyshev_series_roots(
                    beignet.polynomial._chebfromroots.chebfromroots(tgt)
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

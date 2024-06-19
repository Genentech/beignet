import beignet.polynomial
import beignet.polynomial._polyfromroots
import beignet.polynomial._polytrim
import numpy

from tests.beignet._polynomial.test_polynomial import polynomial_Tlist


def test_polyfromroots():
    res = beignet.polynomial._polyfromroots.polyfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._polytrim.polytrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        tgt = polynomial_Tlist[i]
        res = beignet.polynomial._polyfromroots.polyfromroots(roots) * 2 ** (i - 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._polytrim.polytrim(res, tolerance=1e-6),
            beignet.polynomial._polytrim.polytrim(tgt, tolerance=1e-6),
        )

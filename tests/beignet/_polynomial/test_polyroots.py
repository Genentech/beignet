import beignet.polynomial
import beignet.polynomial._polyfromroots
import beignet.polynomial._polyroots
import beignet.polynomial._polytrim
import numpy


def test_polyroots():
    numpy.testing.assert_almost_equal(beignet.polynomial._polyroots.polyroots([1]), [])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._polyroots.polyroots([1, 2]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial._polyroots.polyroots(
            beignet.polynomial._polyfromroots.polyfromroots(tgt)
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._polytrim.polytrim(res, tolerance=1e-6),
            beignet.polynomial._polytrim.polytrim(tgt, tolerance=1e-6),
        )

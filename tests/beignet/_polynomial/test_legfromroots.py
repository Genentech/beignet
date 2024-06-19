import beignet.polynomial
import beignet.polynomial._legtrim
import numpy


def test_legfromroots():
    res = beignet.polynomial.legfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.legfromroots(roots)
        res = beignet.polynomial.legval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.leg2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(res, tgt)

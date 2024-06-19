import beignet.polynomial
import beignet.polynomial._leg2poly
import beignet.polynomial._legfromroots
import beignet.polynomial._legtrim
import beignet.polynomial._legval
import numpy


def test_legfromroots():
    res = beignet.polynomial._legfromroots.legfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._legfromroots.legfromroots(roots)
        res = beignet.polynomial._legval.legval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._leg2poly.leg2poly(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(res, tgt)

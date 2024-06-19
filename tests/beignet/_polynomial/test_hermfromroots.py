import beignet.polynomial
import beignet.polynomial._hermtrim
import numpy


def test_hermfromroots():
    res = beignet.polynomial.hermfromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermtrim.hermtrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.hermfromroots(roots)
        res = beignet.polynomial.hermval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.herm2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(res, tgt)

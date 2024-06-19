import beignet.polynomial
import beignet.polynomial._herme2poly
import beignet.polynomial._hermefromroots
import beignet.polynomial._hermetrim
import beignet.polynomial._hermeval
import numpy


def test_hermefromroots():
    res = beignet.polynomial._hermefromroots.hermefromroots([])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermetrim.hermetrim(res, tolerance=1e-6), [1]
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermefromroots.hermefromroots(roots)
        res = beignet.polynomial._hermeval.hermeval(roots, pol)
        tgt = 0
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._herme2poly.herme2poly(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(res, tgt)

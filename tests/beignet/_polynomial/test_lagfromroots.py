import beignet.polynomial
import beignet.polynomial._lag2poly
import beignet.polynomial._lagfromroots
import beignet.polynomial._lagtrim
import beignet.polynomial._lagval
import numpy


def test_lagfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagtrim.lagtrim(
            beignet.polynomial._lagfromroots.lagfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._lagfromroots.lagfromroots(roots)
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lag2poly.lag2poly(pol)[-1], 1
        )
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagval.lagval(roots, pol), 0
        )

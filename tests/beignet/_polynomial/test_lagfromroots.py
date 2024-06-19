import beignet.polynomial
import beignet.polynomial._lagtrim
import numpy


def test_lagfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._lagtrim.lagtrim(
            beignet.polynomial.lagfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.lagfromroots(roots)
        numpy.testing.assert_(len(pol) == i + 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.lag2poly(pol)[-1], 1)
        numpy.testing.assert_almost_equal(beignet.polynomial.lagval(roots, pol), 0)

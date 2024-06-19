import beignet.polynomial
import beignet.polynomial._lagtrim
import numpy


def test_lagroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.lagroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.lagroots([0, 1]), [1])
    for i in range(2, 5):
        tgt = numpy.linspace(0, 3, i)
        res = beignet.polynomial.lagroots(beignet.polynomial.lagfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial._lagtrim.lagtrim(res, tolerance=1e-6),
            beignet.polynomial._lagtrim.lagtrim(tgt, tolerance=1e-6),
        )

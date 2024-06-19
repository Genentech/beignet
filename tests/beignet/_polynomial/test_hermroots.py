import beignet.polynomial
import numpy


def test_hermroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.hermroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermroots([1, 1]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.hermroots(beignet.polynomial.hermfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial.hermtrim(res, tolerance=1e-6),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )

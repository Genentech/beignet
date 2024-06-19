import beignet.polynomial
import numpy


def test_legtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.legtrim, coef, -1)

    numpy.testing.assert_equal(beignet.polynomial.legtrim(coef), coef[:-1])
    numpy.testing.assert_equal(beignet.polynomial.legtrim(coef, 1), coef[:-3])
    numpy.testing.assert_equal(beignet.polynomial.legtrim(coef, 2), [0])

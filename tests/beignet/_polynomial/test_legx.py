import beignet.polynomial
import numpy


def test_legx():
    numpy.testing.assert_equal(beignet.polynomial.legx, [0, 1])

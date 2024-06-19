import beignet.polynomial
import numpy


def test_legone():
    numpy.testing.assert_equal(beignet.polynomial.legone, [1])

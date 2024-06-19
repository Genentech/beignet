import beignet.polynomial
import numpy


def test_polyx():
    numpy.testing.assert_equal(beignet.polynomial.polyx, [0, 1])

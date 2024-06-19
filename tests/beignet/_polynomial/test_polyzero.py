import beignet.polynomial
import numpy


def test_polyzero():
    numpy.testing.assert_equal(beignet.polynomial.polyzero, [0])

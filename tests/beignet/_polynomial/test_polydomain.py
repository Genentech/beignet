import beignet.polynomial
import numpy


def test_polydomain():
    numpy.testing.assert_equal(beignet.polynomial.polydomain, [-1, 1])

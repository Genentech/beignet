import beignet.polynomial
import numpy


def test_polyone():
    numpy.testing.assert_equal(beignet.polynomial.polyone, [1])

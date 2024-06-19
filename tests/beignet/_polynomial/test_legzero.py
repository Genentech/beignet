import beignet.polynomial
import numpy


def test_legzero():
    numpy.testing.assert_equal(beignet.polynomial.legzero, [0])

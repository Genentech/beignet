import beignet.polynomial
import numpy


def test_polyline():
    numpy.testing.assert_equal(beignet.polynomial.polyline(3, 4), [3, 4])
    numpy.testing.assert_equal(beignet.polynomial.polyline(3, 0), [3])

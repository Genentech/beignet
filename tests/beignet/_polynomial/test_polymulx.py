import beignet.polynomial
import numpy


def test_polymulx():
    numpy.testing.assert_equal(beignet.polynomial.polymulx([0]), [0])
    numpy.testing.assert_equal(beignet.polynomial.polymulx([1]), [0, 1])
    for i in range(1, 5):
        numpy.testing.assert_equal(
            beignet.polynomial.polymulx([0] * i + [1]), [0] * (i + 1) + [1]
        )

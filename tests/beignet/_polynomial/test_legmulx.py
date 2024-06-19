import beignet.polynomial
import numpy


def test_legmulx():
    numpy.testing.assert_equal(beignet.polynomial.legmulx([0]), [0])
    numpy.testing.assert_equal(beignet.polynomial.legmulx([1]), [0, 1])
    for i in range(1, 5):
        tmp = 2 * i + 1
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
        numpy.testing.assert_equal(beignet.polynomial.legmulx(ser), tgt)

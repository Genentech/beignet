import beignet.polynomial
import numpy


def test_legsub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = beignet.polynomial.legsub([0] * i + [1], [0] * j + [1])
            numpy.testing.assert_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

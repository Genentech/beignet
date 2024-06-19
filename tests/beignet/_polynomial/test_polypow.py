import functools

import beignet.polynomial
import numpy


def test_polypow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.polymul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.polypow(c, j)
            numpy.testing.assert_equal(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

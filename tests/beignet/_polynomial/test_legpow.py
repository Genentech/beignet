import functools

import beignet.polynomial
import numpy


def test_legpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.polynomial.legmul, [c] * j, numpy.array([1]))
            res = beignet.polynomial.legpow(c, j)
            numpy.testing.assert_equal(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

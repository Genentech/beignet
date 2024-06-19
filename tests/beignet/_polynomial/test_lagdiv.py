import beignet.polynomial
import numpy


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.lagadd(ci, cj)
            quo, rem = beignet.polynomial.lagdiv(tgt, ci)
            res = beignet.polynomial.lagadd(beignet.polynomial.lagmul(quo, ci), rem)
            numpy.testing.assert_almost_equal(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

import beignet.polynomial
import beignet.polynomial._lagadd
import beignet.polynomial._lagdiv
import beignet.polynomial._lagmul
import beignet.polynomial._lagtrim
import numpy


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._lagadd.lagadd(ci, cj)
            quo, rem = beignet.polynomial._lagdiv.lagdiv(tgt, ci)
            res = beignet.polynomial._lagadd.lagadd(
                beignet.polynomial._lagmul.lagmul(quo, ci), rem
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.lagtrim(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

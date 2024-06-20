import beignet.polynomial
import beignet.polynomial._add_laguerre_series
import beignet.polynomial._divide_laguerre_series
import beignet.polynomial._multiply_laguerre_series
import beignet.polynomial._trim_laguerre_series
import numpy


def test_lagdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._lagadd.add_laguerre_series(ci, cj)
            quo, rem = beignet.polynomial._lagdiv.divide_laguerre_series(tgt, ci)
            res = beignet.polynomial._lagadd.add_laguerre_series(
                beignet.polynomial._lagmul.multiply_laguerre_series(quo, ci), rem
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
                err_msg=msg,
            )

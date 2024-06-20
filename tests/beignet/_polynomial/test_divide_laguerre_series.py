import beignet.polynomial
import numpy


def test_divide_laguerre_series():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.add_laguerre_series(ci, cj)
            quo, rem = beignet.polynomial.divide_laguerre_series(tgt, ci)
            res = beignet.polynomial.add_laguerre_series(
                beignet.polynomial.multiply_laguerre_series(quo, ci), rem
            )
            numpy.testing.assert_almost_equal(
                beignet.polynomial.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial.trim_laguerre_series(tgt, tolerance=1e-6),
                err_msg=msg,
            )

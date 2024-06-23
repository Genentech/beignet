import functools

import beignet.polynomial
import beignet.polynomial._multiply_laguerre_series
import beignet.polynomial._pow_laguerre_series
import beignet.polynomial._trim_laguerre_series
import numpy
import torch


def test_lagpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._lagmul.multiply_laguerre_series,
                [c] * j,
                numpy.array([1]),
            )
            res = beignet.polynomial._lagpow.pow_laguerre_series(c, j)
            torch.testing.assert_close(
                beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
                err_msg=msg,
            )

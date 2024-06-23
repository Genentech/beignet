import functools

import beignet.polynomial
import numpy
import torch


def test_pow_laguerre_series():
    for i in range(5):
        for j in range(5):
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.multiply_laguerre_series,
                [c] * j,
                numpy.array([1]),
            )
            res = beignet.polynomial.pow_laguerre_series(c, j)
            torch.testing.assert_close(
                beignet.polynomial.trim_laguerre_series(res, tolerance=1e-6),
                beignet.polynomial.trim_laguerre_series(tgt, tolerance=1e-6),
            )

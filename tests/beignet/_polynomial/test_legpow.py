import functools

import beignet.polynomial
import beignet.polynomial._multiply_legendre_series
import beignet.polynomial._pow_legendre_series
import beignet.polynomial._trim_legendre_series
import numpy
import torch


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._legmul.multiply_legendre_series,
                [c] * j,
                numpy.array([1]),
            )
            res = beignet.polynomial._legpow.pow_legendre_series(c, j)
            torch.testing.assert_close(
                beignet.polynomial._legtrim.trim_legendre_series(res, tolerance=1e-6),
                beignet.polynomial._legtrim.trim_legendre_series(tgt, tolerance=1e-6),
            )

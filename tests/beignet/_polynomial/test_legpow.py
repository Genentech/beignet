import functools

import beignet.polynomial
import numpy
import torch


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.polynomial.legmul, [c] * j, numpy.array([1]))
            res = beignet.polynomial.legpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
            )

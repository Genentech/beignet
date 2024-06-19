import functools

import beignet.polynomial
import beignet.polynomial._legmul
import beignet.polynomial._legpow
import beignet.polynomial._legtrim
import numpy
import torch


def test_legpow():
    for i in range(5):
        for j in range(5):
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._legmul.legmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial._legpow.legpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
            )

import functools

import beignet.polynomial
import beignet.polynomial._chebtrim
import numpy
import torch


def test_chebpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.chebmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.chebpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

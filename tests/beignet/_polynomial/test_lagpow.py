import functools

import beignet.polynomial
import numpy
import torch


def test_lagpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(beignet.polynomial.lagmul, [c] * j, numpy.array([1]))
            res = beignet.polynomial.lagpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(res, tolerance=1e-6),
                beignet.polynomial.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

import functools

import beignet.polynomial
import numpy
import torch


def test_hermpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.hermmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial.hermpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

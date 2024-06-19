import functools

import beignet.polynomial
import beignet.polynomial._hermmul
import beignet.polynomial._hermpow
import beignet.polynomial._hermtrim
import numpy
import torch


def test_hermpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._hermmul.hermmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial._hermpow.hermpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial._hermtrim.hermtrim(res, tolerance=1e-6),
                beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

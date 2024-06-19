import functools

import beignet.polynomial
import beignet.polynomial._hermemul
import beignet.polynomial._hermepow
import beignet.polynomial._hermetrim
import numpy
import torch


def test_hermepow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._hermemul.hermemul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial._hermepow.hermepow(c, j)
            torch.testing.assert_close(
                beignet.polynomial._hermetrim.hermetrim(res, tolerance=1e-6),
                beignet.polynomial._hermetrim.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

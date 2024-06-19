import functools

import beignet.polynomial
import beignet.polynomial._lagmul
import beignet.polynomial._lagpow
import beignet.polynomial._lagtrim
import numpy
import torch


def test_lagpow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._lagmul.lagmul, [c] * j, numpy.array([1])
            )
            res = beignet.polynomial._lagpow.lagpow(c, j)
            torch.testing.assert_close(
                beignet.polynomial._lagtrim.lagtrim(res, tolerance=1e-6),
                beignet.polynomial._lagtrim.lagtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

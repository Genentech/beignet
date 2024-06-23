import functools

import beignet.polynomial
import beignet.polynomial._multiply_probabilists_hermite_series
import beignet.polynomial._pow_probabilists_hermite_series
import beignet.polynomial._trim_probabilists_hermite_series
import numpy
import torch


def test_hermepow():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial._hermemul.multiply_probabilists_hermite_series,
                [c] * j,
                numpy.array([1]),
            )
            res = beignet.polynomial._hermepow.pow_probabilists_hermite_series(c, j)
            torch.testing.assert_close(
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
                    tgt, tolerance=1e-6
                ),
                err_msg=msg,
            )

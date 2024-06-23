import functools

import beignet.polynomial
import numpy
import torch


def test_pow_physicists_hermite_series():
    for i in range(5):
        for j in range(5):
            c = numpy.arange(i + 1)
            tgt = functools.reduce(
                beignet.polynomial.multiply_physicists_hermite_series,
                [c] * j,
                numpy.array([1]),
            )
            res = beignet.polynomial.pow_physicists_hermite_series(c, j)
            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(res, tolerance=1e-6),
                beignet.polynomial.trim_physicists_hermite_series(tgt, tolerance=1e-6),
            )

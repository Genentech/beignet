import beignet.polynomial
import numpy
import torch


def test_polymul():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(i + j + 1)
            tgt[i + j] += 1
            res = beignet.polynomial.polymul([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

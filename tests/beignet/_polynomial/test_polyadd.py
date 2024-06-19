import beignet.polynomial
import numpy
import torch


def test_polyadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.polyadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.polytrim(res, tolerance=1e-6),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

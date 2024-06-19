import beignet.polynomial
import numpy
import torch


def test_hermeadd():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = beignet.polynomial.hermeadd([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

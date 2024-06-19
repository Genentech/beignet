import beignet.polynomial
import numpy
import torch


def test_polysub():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polysub([0] * i + [1], [0] * j + [1]),
                    tolerance=1e-6,
                ),
                beignet.polynomial.polytrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

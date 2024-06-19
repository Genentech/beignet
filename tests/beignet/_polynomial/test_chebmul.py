import beignet.polynomial
import beignet.polynomial._chebtrim
import numpy
import torch


def test_chebmul():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            tgt = numpy.zeros(i + j + 1)
            tgt[i + j] += 0.5
            tgt[abs(i - j)] += 0.5
            res = beignet.polynomial.chebmul([0] * i + [1], [0] * j + [1])
            torch.testing.assert_close(
                beignet.polynomial._chebtrim.chebtrim(res, tolerance=1e-6),
                beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

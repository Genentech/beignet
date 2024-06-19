import beignet.polynomial
import beignet.polynomial._chebadd
import torch


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            tgt = beignet.polynomial._chebadd.chebadd(ci, [0] * j + [1])
            quo, rem = beignet.polynomial.chebdiv(tgt, ci)
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial._chebadd.chebadd(
                        beignet.polynomial.chebmul(quo, ci), rem
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

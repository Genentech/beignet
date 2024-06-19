import beignet.polynomial
import beignet.polynomial._legadd
import torch


def test_legdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._legadd.legadd(ci, cj)
            quo, rem = beignet.polynomial.legdiv(tgt, ci)
            res = beignet.polynomial._legadd.legadd(
                beignet.polynomial.legmul(quo, ci), rem
            )
            torch.testing.assert_close(
                beignet.polynomial.legtrim(res, tolerance=1e-6),
                beignet.polynomial.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

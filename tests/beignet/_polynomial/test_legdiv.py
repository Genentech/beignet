import beignet.polynomial
import beignet.polynomial._legadd
import beignet.polynomial._legdiv
import beignet.polynomial._legmul
import beignet.polynomial._legtrim
import torch


def test_legdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._legadd.legadd(ci, cj)
            quo, rem = beignet.polynomial._legdiv.legdiv(tgt, ci)
            res = beignet.polynomial._legadd.legadd(
                beignet.polynomial._legmul.legmul(quo, ci), rem
            )
            torch.testing.assert_close(
                beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
                beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

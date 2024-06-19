import beignet.polynomial
import beignet.polynomial._hermadd
import beignet.polynomial._hermtrim
import torch


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._hermadd.hermadd(ci, cj)
            quo, rem = beignet.polynomial.hermdiv(tgt, ci)
            res = beignet.polynomial._hermadd.hermadd(
                beignet.polynomial.hermmul(quo, ci), rem
            )
            torch.testing.assert_close(
                beignet.polynomial._hermtrim.hermtrim(res, tolerance=1e-6),
                beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

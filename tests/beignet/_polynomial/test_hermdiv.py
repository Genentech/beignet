import beignet.polynomial
import torch


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.hermadd(ci, cj)
            quo, rem = beignet.polynomial.hermdiv(tgt, ci)
            res = beignet.polynomial.hermadd(beignet.polynomial.hermmul(quo, ci), rem)
            torch.testing.assert_close(
                beignet.polynomial.hermtrim(res, tolerance=1e-6),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

import beignet.polynomial
import beignet.polynomial._hermeadd
import torch


def test_hermediv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._hermeadd.hermeadd(ci, cj)
            quo, rem = beignet.polynomial.hermediv(tgt, ci)
            res = beignet.polynomial._hermeadd.hermeadd(
                beignet.polynomial.hermemul(quo, ci), rem
            )
            torch.testing.assert_close(
                beignet.polynomial.hermetrim(res, tolerance=1e-6),
                beignet.polynomial.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

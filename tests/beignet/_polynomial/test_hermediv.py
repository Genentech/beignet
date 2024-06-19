import beignet.polynomial
import beignet.polynomial._hermeadd
import beignet.polynomial._hermediv
import beignet.polynomial._hermemul
import beignet.polynomial._hermetrim
import torch


def test_hermediv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._hermeadd.hermeadd(ci, cj)
            quo, rem = beignet.polynomial._hermediv.hermediv(tgt, ci)
            res = beignet.polynomial._hermeadd.hermeadd(
                beignet.polynomial._hermemul.hermemul(quo, ci), rem
            )
            torch.testing.assert_close(
                beignet.polynomial._hermetrim.hermetrim(res, tolerance=1e-6),
                beignet.polynomial._hermetrim.hermetrim(tgt, tolerance=1e-6),
                err_msg=msg,
            )

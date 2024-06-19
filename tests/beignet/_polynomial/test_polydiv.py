import beignet.polynomial
import beignet.polynomial._polyadd
import beignet.polynomial._polydiv
import beignet.polynomial._polymul
import numpy
import torch


def test_polydiv():
    numpy.testing.assert_raises(
        ZeroDivisionError, beignet.polynomial._polydiv.polydiv, [1], [0]
    )

    quo, rem = beignet.polynomial._polydiv.polydiv([2], [2])
    torch.testing.assert_close((quo, rem), (1, 0))
    quo, rem = beignet.polynomial._polydiv.polydiv([2, 2], [2])
    torch.testing.assert_close((quo, rem), ((1, 1), 0))

    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1, 2]
            cj = [0] * j + [1, 2]
            tgt = beignet.polynomial._polyadd.polyadd(ci, cj)
            quo, rem = beignet.polynomial._polydiv.polydiv(tgt, ci)
            torch.testing.assert_close(
                beignet.polynomial._polyadd.polyadd(
                    beignet.polynomial._polymul.polymul(quo, ci), rem
                ),
                tgt,
                err_msg=msg,
            )

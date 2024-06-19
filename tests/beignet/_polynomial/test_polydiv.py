import beignet.polynomial
import beignet.polynomial._polyadd
import numpy
import torch


def test_polydiv():
    numpy.testing.assert_raises(ZeroDivisionError, beignet.polynomial.polydiv, [1], [0])

    quo, rem = beignet.polynomial.polydiv([2], [2])
    torch.testing.assert_close((quo, rem), (1, 0))
    quo, rem = beignet.polynomial.polydiv([2, 2], [2])
    torch.testing.assert_close((quo, rem), ((1, 1), 0))

    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1, 2]
            cj = [0] * j + [1, 2]
            tgt = beignet.polynomial._polyadd.polyadd(ci, cj)
            quo, rem = beignet.polynomial.polydiv(tgt, ci)
            torch.testing.assert_close(
                beignet.polynomial._polyadd.polyadd(
                    beignet.polynomial.polymul(quo, ci), rem
                ),
                tgt,
                err_msg=msg,
            )

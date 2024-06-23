import beignet.polynomial
import numpy
import torch


def test_laguerre_series_roots():
    torch.testing.assert_close(
        beignet.polynomial._lagroots.laguerre_series_roots([1]), []
    )
    torch.testing.assert_close(
        beignet.polynomial._lagroots.laguerre_series_roots([0, 1]), [1]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(0, 3, i)
        res = beignet.polynomial._lagroots.laguerre_series_roots(
            beignet.polynomial._lagfromroots.lagfromroots(tgt)
        )
        torch.testing.assert_close(
            beignet.polynomial._lagtrim.trim_laguerre_series(res, tolerance=1e-6),
            beignet.polynomial._lagtrim.trim_laguerre_series(tgt, tolerance=1e-6),
        )

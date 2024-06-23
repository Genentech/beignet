import beignet.polynomial
import beignet.polynomial._hermfromroots
import beignet.polynomial._physicists_hermite_series_roots
import beignet.polynomial._trim_physicists_hermite_series
import numpy
import torch


def test_physicists_hermite_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_roots([1]), []
    )
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_roots([1, 1]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.physicists_hermite_series_roots(
            beignet.polynomial._hermfromroots.hermfromroots(tgt)
        )
        torch.testing.assert_close(
            beignet.polynomial.trim_physicists_hermite_series(res, tolerance=1e-6),
            beignet.polynomial.trim_physicists_hermite_series(tgt, tolerance=1e-6),
        )

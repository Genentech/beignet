import beignet.polynomial
import numpy
import torch


def test_hermfromroots():
    res = beignet.polynomial._hermfromroots.hermfromroots([])
    torch.testing.assert_close(
        beignet.polynomial._hermtrim.trim_physicists_hermite_series(
            res, tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermfromroots.hermfromroots(roots)
        res = beignet.polynomial._hermval.evaluate_physicists_hermite_series_1d(
            roots, pol
        )
        tgt = 0
        assert len(pol) == i + 1
        torch.testing.assert_close(
            beignet.polynomial._herm2poly.physicists_hermite_series_to_power_series(
                pol
            )[-1],
            1,
        )
        torch.testing.assert_close(res, tgt)

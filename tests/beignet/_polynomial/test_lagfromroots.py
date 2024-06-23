import beignet.polynomial
import numpy
import torch


def test_lagfromroots():
    torch.testing.assert_close(
        beignet.polynomial._lagtrim.trim_laguerre_series(
            beignet.polynomial._lagfromroots.lagfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._lagfromroots.lagfromroots(roots)
        assert len(pol) == i + 1
        torch.testing.assert_close(
            beignet.polynomial._lag2poly.laguerre_series_to_power_series(pol)[-1], 1
        )
        torch.testing.assert_close(
            beignet.polynomial._lagval.evaluate_laguerre_series_1d(roots, pol), 0
        )

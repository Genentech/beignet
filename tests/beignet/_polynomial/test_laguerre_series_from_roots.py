import beignet.polynomial
import numpy
import torch


def test_laguerre_series_from_roots():
    torch.testing.assert_close(
        beignet.polynomial.trim_laguerre_series(
            beignet.polynomial._lagfromroots.laguerre_series_from_roots([]),
            tolerance=0.000001,
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._lagfromroots.laguerre_series_from_roots(roots)
        assert len(pol) == i + 1
        torch.testing.assert_close(
            beignet.polynomial.laguerre_series_to_power_series(pol)[-1], 1
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_laguerre_series_1d(roots, pol), 0
        )

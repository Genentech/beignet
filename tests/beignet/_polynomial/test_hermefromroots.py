import beignet.polynomial
import numpy
import torch


def test_hermefromroots():
    res = beignet.polynomial._hermefromroots.probabilists_hermite_series_from_roots([])
    torch.testing.assert_close(
        beignet.polynomial._hermetrim.trim_probabilists_hermite_series(
            res, tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial._hermefromroots.probabilists_hermite_series_from_roots(
            roots
        )
        res = beignet.polynomial._hermeval.evaluate_probabilists_hermite_series_1d(
            roots, pol
        )
        tgt = 0
        assert len(pol) == i + 1
        torch.testing.assert_close(
            beignet.polynomial._herme2poly.probabilists_hermite_series_to_power_series(
                pol
            )[-1],
            1,
        )
        torch.testing.assert_close(res, tgt)

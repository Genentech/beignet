import beignet.polynomial
import numpy
import torch


def test_probabilists_hermite_series_from_roots():
    res = beignet.polynomial.probabilists_hermite_series_from_roots([])
    torch.testing.assert_close(
        beignet.polynomial.trim_probabilists_hermite_series(res, tolerance=1e-6),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        pol = beignet.polynomial.probabilists_hermite_series_from_roots(roots)
        res = beignet.polynomial.evaluate_probabilists_hermite_series_1d(roots, pol)
        tgt = 0
        assert len(pol) == i + 1
        torch.testing.assert_close(
            beignet.polynomial.probabilists_hermite_series_to_power_series(pol)[-1],
            1,
        )
        torch.testing.assert_close(res, tgt)

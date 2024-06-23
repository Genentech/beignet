import beignet.polynomial
import numpy
import torch


def test_evaluate_power_series_3d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises_regex(
        ValueError,
        "incompatible",
        beignet.polynomial.evaluate_power_series_3d,
        x1,
        x2,
        x3[:2],
        c3d,
    )

    tgt = y1 * y2 * y3
    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_3d(x1, x2, x3, c3d), tgt
    )

    z = numpy.ones((2, 3))
    assert beignet.polynomial.evaluate_power_series_3d(z, z, z, c3d).shape == (2, 3)

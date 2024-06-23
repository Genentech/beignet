import beignet.polynomial
import numpy
import torch.testing


def test_evaluate_power_series_2d():
    c1d = numpy.array([1.0, 2.0, 3.0])

    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1

    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises_regex(
        ValueError,
        "incompatible",
        beignet.polynomial.evaluate_power_series_2d,
        x1,
        x2[:2],
        c2d,
    )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_2d(x1, x2, c2d),
        y1 * y2,
    )

    z = numpy.ones((2, 3))

    assert beignet.polynomial.evaluate_power_series_2d(z, z, c2d).shape == (2, 3)

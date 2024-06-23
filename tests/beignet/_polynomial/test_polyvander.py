import beignet.polynomial
import numpy
import torch.testing


def test_polyvander():
    x = numpy.arange(3)
    v = beignet.polynomial.power_series_vandermonde_1d(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_power_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])

    v = beignet.polynomial.power_series_vandermonde_1d(x, 3)

    assert v.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_power_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
        )

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.power_series_vandermonde_1d, numpy.arange(3), -1
    )

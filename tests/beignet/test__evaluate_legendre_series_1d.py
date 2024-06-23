import math

import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_1d
import beignet.polynomial._evaluate_power_series_1d
import torch


def test_evaluate_legendre_series_1d():
    output = beignet.polynomial.evaluate_legendre_series_1d(
        torch.tensor([]),
        torch.tensor([1]),
    )

    assert math.prod(output.shape) == 0

    # x = numpy.linspace(-1, 1)
    #
    # y = []
    #
    # for c in legendre_polynomial_coefficients:
    #     y.append(beignet.polynomial.evaluate_power_series_1d(x, c))
    #
    # for index in range(10):
    #     torch.testing.assert_close(
    #         beignet.polynomial.evaluate_legendre_series_1d(x, [0] * index + [1]),
    #         y[index],
    #     )
    #
    # for index in range(3):
    #     x = torch.zeros([2] * index)
    #
    #     output = beignet.polynomial.evaluate_legendre_series_1d(
    #         x,
    #         torch.tensor([1]),
    #     )
    #
    #     assert output.shape == [2] * index
    #
    #     output = beignet.polynomial.evaluate_legendre_series_1d(
    #         x,
    #         torch.tensor([1, 0]),
    #     )
    #
    #     assert output.shape == [2] * index
    #
    #     output = beignet.polynomial.evaluate_legendre_series_1d(
    #         x,
    #         torch.tensor([1, 0, 0]),
    #     )
    #
    #     assert output.shape == [2] * index

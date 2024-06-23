import math

import beignet.polynomial
import torch

chebyshev_polynomial_coefficients = [
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.tensor([-1, 0, 2]),
    torch.tensor([0, -3, 0, 4]),
    torch.tensor([1, 0, -8, 0, 8]),
    torch.tensor([0, 5, 0, -20, 0, 16]),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32]),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]


def test_evaluate_chebyshev_series_1d():
    output = beignet.polynomial.evaluate_chebyshev_series_1d(
        torch.tensor([]),
        torch.tensor([1]),
    )

    assert math.prod(output.shape) == 0

    x = torch.linspace(-1, 1, 50)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in chebyshev_polynomial_coefficients
    ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
            y[index],
            atol=1e-5,
            rtol=1e-5,
        )

    for index in range(3):
        output = beignet.polynomial.evaluate_chebyshev_series_1d(
            torch.zeros([2] * index),
            torch.tensor([1]),
        )

        assert output.shape == [2] * index

        output = beignet.polynomial.evaluate_chebyshev_series_1d(
            torch.zeros([2] * index),
            torch.tensor([1, 0]),
        )

        assert output.shape == [2] * index

        output = beignet.polynomial.evaluate_chebyshev_series_1d(
            torch.zeros([2] * index),
            torch.tensor([1, 0, 0]),
        )

        assert output.shape == [2] * index

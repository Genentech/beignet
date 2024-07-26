import math

import beignet
import torch


def test_evaluate_legendre_polynomial():
    coefficients = [
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 0, 3]) / 2,
        torch.tensor([0, -3, 0, 5]) / 2,
        torch.tensor([3, 0, -30, 0, 35]) / 8,
        torch.tensor([0, 15, 0, -70, 0, 63]) / 8,
        torch.tensor([-5, 0, 105, 0, -315, 0, 231]) / 16,
        torch.tensor([0, -35, 0, 315, 0, -693, 0, 429]) / 16,
        torch.tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        torch.tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    output = beignet.evaluate_legendre_polynomial(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in coefficients:
        ys = [
            *ys,
            beignet.evaluate_polynomial(
                torch.linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for i in range(10):
        torch.testing.assert_close(
            beignet.evaluate_legendre_polynomial(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor(ys[i]),
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = beignet.evaluate_legendre_polynomial(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_legendre_polynomial(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_legendre_polynomial(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

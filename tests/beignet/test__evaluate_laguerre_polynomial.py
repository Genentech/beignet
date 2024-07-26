import math

import beignet
import torch


def test_evaluate_laguerre_polynomial():
    coefficients = [
        torch.tensor([1]) / 1,
        torch.tensor([1, -1]) / 1,
        torch.tensor([2, -4, 1]) / 2,
        torch.tensor([6, -18, 9, -1]) / 6,
        torch.tensor([24, -96, 72, -16, 1]) / 24,
        torch.tensor([120, -600, 600, -200, 25, -1]) / 120,
        torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720,
    ]

    output = beignet.evaluate_laguerre_polynomial(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    input = torch.linspace(-1, 1, 50)

    for coefficient in coefficients:
        ys = [
            *ys,
            beignet.evaluate_polynomial(
                input,
                coefficient,
            ),
        ]

    for i in range(7):
        torch.testing.assert_close(
            beignet.evaluate_laguerre_polynomial(
                input,
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor(torch.tensor(ys[i])),
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = beignet.evaluate_laguerre_polynomial(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_laguerre_polynomial(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_laguerre_polynomial(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

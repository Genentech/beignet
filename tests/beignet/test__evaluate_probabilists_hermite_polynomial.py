import math

import beignet
import torch


def test_evaluate_probabilists_hermite_polynomial():
    coefficients = [
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, -3, 0, 1]),
        torch.tensor([3, 0, -6, 0, 1]),
        torch.tensor([0, 15, 0, -10, 0, 1]),
        torch.tensor([-15, 0, 45, 0, -15, 0, 1]),
        torch.tensor([0, -105, 0, 105, 0, -21, 0, 1]),
        torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1]),
        torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    output = beignet.evaluate_probabilists_hermite_polynomial(
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
            beignet.evaluate_probabilists_hermite_polynomial(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * i + [1.0]),
            ),
            ys[i],
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = beignet.evaluate_probabilists_hermite_polynomial(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_probabilists_hermite_polynomial(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_probabilists_hermite_polynomial(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

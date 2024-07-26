import math

import beignet
import torch


def test_evaluate_physicists_hermite_polynomial():
    coefficients = [
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([-2, 0, 4]),
        torch.tensor([0, -12, 0, 8]),
        torch.tensor([12, 0, -48, 0, 16]),
        torch.tensor([0, 120, 0, -160, 0, 32]),
        torch.tensor([-120, 0, 720, 0, -480, 0, 64]),
        torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128]),
        torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        torch.tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    output = beignet.evaluate_physicists_hermite_polynomial(
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

    for index in range(10):
        torch.testing.assert_close(
            beignet.evaluate_physicists_hermite_polynomial(
                input,
                torch.tensor([0.0] * index + [1.0]),
            ),
            ys[index],
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = beignet.evaluate_physicists_hermite_polynomial(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_physicists_hermite_polynomial(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_physicists_hermite_polynomial(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

import math

import torch

import beignet.polynomials


def test_evaluate_chebyshev_polynomial(float64):
    chebcoefficients = [
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

    output = beignet.polynomials.evaluate_chebyshev_polynomial(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    ys = []

    for coefficient in chebcoefficients:
        ys = [
            *ys,
            beignet.polynomials.evaluate_polynomial(
                torch.linspace(-1, 1, 50),
                coefficient,
            ),
        ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomials.evaluate_chebyshev_polynomial(
                torch.linspace(-1, 1, 50),
                torch.tensor([0.0] * index + [1.0]),
            ),
            ys[index],
        )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = beignet.polynomials.evaluate_chebyshev_polynomial(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.polynomials.evaluate_chebyshev_polynomial(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.polynomials.evaluate_chebyshev_polynomial(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

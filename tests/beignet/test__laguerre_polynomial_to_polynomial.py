import beignet
import torch


def test_laguerre_polynomial_to_polynomial():
    coefficients = [
        torch.tensor([1.0]) / 1,
        torch.tensor([1.0, -1]) / 1,
        torch.tensor([2.0, -4, 1]) / 2,
        torch.tensor([6.0, -18, 9, -1]) / 6,
        torch.tensor([24.0, -96, 72, -16, 1]) / 24,
        torch.tensor([120.0, -600, 600, -200, 25, -1]) / 120,
        torch.tensor([720.0, -4320, 5400, -2400, 450, -36, 1]) / 720,
    ]

    for index in range(7):
        torch.testing.assert_close(
            beignet.laguerre_polynomial_to_polynomial(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )

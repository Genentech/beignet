import beignet
import torch


def test_polynomial_to_laguerre_polynomial():
    coefficients = [
        torch.tensor([1.0]) / 1.0,
        torch.tensor([1.0, -1.0]) / 1.0,
        torch.tensor([2.0, -4.0, 1.0]) / 2.0,
        torch.tensor([6.0, -18.0, 9.0, -1.0]) / 6.0,
        torch.tensor([24.0, -96.0, 72.0, -16.0, 1.0]) / 24.0,
        torch.tensor([120.0, -600.0, 600.0, -200.0, 25.0, -1.0]) / 120.0,
        torch.tensor([720.0, -4320.0, 5400.0, -2400.0, 450.0, -36.0, 1.0]) / 720.0,
    ]

    for index in range(7):
        torch.testing.assert_close(
            beignet.polynomial_to_laguerre_polynomial(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )

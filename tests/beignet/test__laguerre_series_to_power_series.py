import beignet.polynomial
import torch

laguerre_polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float64) / 1,
    torch.tensor([1, -1], dtype=torch.float64) / 1,
    torch.tensor([2, -4, 1], dtype=torch.float64) / 2,
    torch.tensor([6, -18, 9, -1], dtype=torch.float64) / 6,
    torch.tensor([24, -96, 72, -16, 1], dtype=torch.float64) / 24,
    torch.tensor([120, -600, 600, -200, 25, -1], dtype=torch.float64) / 120,
    torch.tensor([720, -4320, 5400, -2400, 450, -36, 1], dtype=torch.float64) / 720,
]


def test_laguerre_series_to_power_series():
    for index in range(7):
        torch.testing.assert_close(
            beignet.polynomial.laguerre_series_to_power_series(
                torch.tensor([0] * index + [1]),
            ),
            laguerre_polynomial_coefficients[index],
        )

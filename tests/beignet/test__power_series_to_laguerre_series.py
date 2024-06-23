import beignet.polynomial
import torch.testing

laguerre_polynomial_coefficients = [
    (torch.tensor([1]) / 1),
    (torch.tensor([1, -1]) / 1),
    (torch.tensor([2, -4, 1]) / 2),
    (torch.tensor([6, -18, 9, -1]) / 6),
    (torch.tensor([24, -96, 72, -16, 1]) / 24),
    (torch.tensor([120, -600, 600, -200, 25, -1]) / 120),
    (torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]


def test_power_series_to_laguerre_series():
    for index in range(7):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_laguerre_series(
                laguerre_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float32),
        )

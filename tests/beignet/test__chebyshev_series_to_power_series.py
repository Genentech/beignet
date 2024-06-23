import beignet.polynomial
import torch

chebyshev_polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float64),
    torch.tensor([0, 1], dtype=torch.float64),
    torch.tensor([-1, 0, 2], dtype=torch.float64),
    torch.tensor([0, -3, 0, 4], dtype=torch.float64),
    torch.tensor([1, 0, -8, 0, 8], dtype=torch.float64),
    torch.tensor([0, 5, 0, -20, 0, 16], dtype=torch.float64),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32], dtype=torch.float64),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64], dtype=torch.float64),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128], dtype=torch.float64),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256], dtype=torch.float64),
]


def test_chebyshev_series_to_power_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.chebyshev_series_to_power_series(
                torch.tensor([0] * index + [1]),
            ),
            chebyshev_polynomial_coefficients[index],
        )

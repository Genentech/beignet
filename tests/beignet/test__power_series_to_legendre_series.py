import beignet.polynomial
import torch.testing

legendre_polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float64),
    torch.tensor([0, 1], dtype=torch.float64),
    torch.tensor([-1, 0, 3], dtype=torch.float64) / 2,
    torch.tensor([0, -3, 0, 5], dtype=torch.float64) / 2,
    torch.tensor([3, 0, -30, 0, 35], dtype=torch.float64) / 8,
    torch.tensor([0, 15, 0, -70, 0, 63], dtype=torch.float64) / 8,
    torch.tensor([-5, 0, 105, 0, -315, 0, 231], dtype=torch.float64) / 16,
    torch.tensor([0, -35, 0, 315, 0, -693, 0, 429], dtype=torch.float64) / 16,
    torch.tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435], dtype=torch.float64)
    / 128,
    torch.tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155], dtype=torch.float64)
    / 128,
]


def test_power_series_to_legendre_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.power_series_to_legendre_series(
                legendre_polynomial_coefficients[index],
            ),
            torch.tensor([0] * index + [1], dtype=torch.float64),
        )

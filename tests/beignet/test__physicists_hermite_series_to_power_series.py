import beignet.polynomial
import torch

hermite_polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float64),
    torch.tensor([0, 2], dtype=torch.float64),
    torch.tensor([-2, 0, 4], dtype=torch.float64),
    torch.tensor([0, -12, 0, 8], dtype=torch.float64),
    torch.tensor([12, 0, -48, 0, 16], dtype=torch.float64),
    torch.tensor([0, 120, 0, -160, 0, 32], dtype=torch.float64),
    torch.tensor([-120, 0, 720, 0, -480, 0, 64], dtype=torch.float64),
    torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128], dtype=torch.float64),
    torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256], dtype=torch.float64),
    torch.tensor(
        [0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512], dtype=torch.float64
    ),
]


def test_physicists_hermite_series_to_power_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.physicists_hermite_series_to_power_series(
                torch.tensor([0] * index + [1]),
            ),
            hermite_polynomial_coefficients[index],
        )

import beignet.polynomial
import torch

hermite_e_polynomial_coefficients = [
    (torch.tensor([1], dtype=torch.float64)),
    (torch.tensor([0, 1], dtype=torch.float64)),
    (torch.tensor([-1, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -3, 0, 1], dtype=torch.float64)),
    (torch.tensor([3, 0, -6, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 15, 0, -10, 0, 1], dtype=torch.float64)),
    (torch.tensor([-15, 0, 45, 0, -15, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -105, 0, 105, 0, -21, 0, 1], dtype=torch.float64)),
    (torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1], dtype=torch.float64)),
]


def test_probabilists_hermite_series_to_power_series():
    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial.probabilists_hermite_series_to_power_series(
                torch.tensor([0] * index + [1])
            ),
            hermite_e_polynomial_coefficients[index],
        )

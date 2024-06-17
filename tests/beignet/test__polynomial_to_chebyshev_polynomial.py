import beignet
import torch


def test_polynomial_to_chebyshev_polynomial():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 2]),
        torch.tensor([0.0, -3, 0, 4]),
        torch.tensor([1.0, 0, -8, 0, 8]),
        torch.tensor([0.0, 5, 0, -20, 0, 16]),
        torch.tensor([-1.0, 0, 18, 0, -48, 0, 32]),
        torch.tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
        torch.tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
        torch.tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial_to_chebyshev_polynomial(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )

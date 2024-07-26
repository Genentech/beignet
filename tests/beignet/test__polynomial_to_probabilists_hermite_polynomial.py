import beignet
import torch


def test_polynomial_to_probabilists_hermite_polynomial():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1]),
        torch.tensor([-1.0, 0, 1]),
        torch.tensor([0.0, -3, 0, 1]),
        torch.tensor([3.0, 0, -6, 0, 1]),
        torch.tensor([0.0, 15, 0, -10, 0, 1]),
        torch.tensor([-15.0, 0, 45, 0, -15, 0, 1]),
        torch.tensor([0.0, -105, 0, 105, 0, -21, 0, 1]),
        torch.tensor([105.0, 0, -420, 0, 210, 0, -28, 0, 1]),
        torch.tensor([0.0, 945, 0, -1260, 0, 378, 0, -36, 0, 1]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.polynomial_to_probabilists_hermite_polynomial(
                coefficients[index],
            ),
            torch.tensor([0.0] * index + [1.0]),
        )

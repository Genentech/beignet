import beignet
import torch


def test_legendre_polynomial_to_polynomial():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0, 3.0]) / 2.0,
        torch.tensor([0.0, -3.0, 0.0, 5.0]) / 2.0,
        torch.tensor([3.0, 0.0, -30, 0, 35]) / 8,
        torch.tensor([0.0, 15.0, 0, -70, 0, 63]) / 8,
        torch.tensor([-5.0, 0.0, 105, 0, -315, 0, 231]) / 16,
        torch.tensor([0.0, -35.0, 0, 315, 0, -693, 0, 429]) / 16,
        torch.tensor([35.0, 0.0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128,
        torch.tensor([0.0, 315.0, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128,
    ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.legendre_polynomial_to_polynomial(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )

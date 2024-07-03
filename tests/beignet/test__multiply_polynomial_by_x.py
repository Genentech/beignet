import beignet
import torch


def test_multiply_polynomial_by_x():
    torch.testing.assert_close(
        beignet.multiply_polynomial_by_x(
            torch.tensor([0.0]),
        ),
        torch.tensor([0.0, 0.0]),
    )

    torch.testing.assert_close(
        beignet.multiply_polynomial_by_x(
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            beignet.multiply_polynomial_by_x(
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor([0.0] * (i + 1) + [1.0]),
        )

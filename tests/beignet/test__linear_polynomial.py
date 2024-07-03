import beignet
import torch


def test_linear_polynomial():
    torch.testing.assert_close(
        beignet.linear_polynomial(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        beignet.linear_polynomial(3.0, 0.0),
        torch.tensor([3.0, 0.0]),
    )

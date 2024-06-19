import beignet.polynomial
import torch


def test_hermline():
    torch.testing.assert_close(
        beignet.polynomial.hermline(3, 4),
        torch.tensor([3, 2]),
    )

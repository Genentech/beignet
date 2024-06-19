import beignet.polynomial
import beignet.polynomial._hermeline
import torch


def test_hermeline():
    torch.testing.assert_close(
        beignet.polynomial._hermeline.hermeline(3, 4),
        torch.tensor([3, 4]),
    )

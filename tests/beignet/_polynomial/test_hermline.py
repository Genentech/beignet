import beignet.polynomial
import beignet.polynomial._hermline
import torch


def test_hermline():
    torch.testing.assert_close(
        beignet.polynomial._hermline.hermline(3, 4),
        torch.tensor([3, 2], dtype=torch.float32),
    )

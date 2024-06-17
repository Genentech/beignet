import beignet
import torch


def test_laguerre_polynomial_x():
    torch.testing.assert_close(
        beignet.laguerre_polynomial_x,
        torch.tensor([1.0, -1.0]),
        check_dtype=False,
    )

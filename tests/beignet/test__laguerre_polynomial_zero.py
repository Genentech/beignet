import beignet
import torch


def test_laguerre_polynomial_zero():
    torch.testing.assert_close(
        beignet.laguerre_polynomial_zero,
        torch.tensor([0.0]),
        check_dtype=False,
    )

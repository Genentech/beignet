import beignet
import torch


def test_polynomial_one():
    torch.testing.assert_close(
        beignet.polynomial_one,
        torch.tensor([1.0]),
        check_dtype=False,
    )

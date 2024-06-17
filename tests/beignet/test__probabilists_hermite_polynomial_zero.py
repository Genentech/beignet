import beignet
import torch


def test_probabilists_hermite_polynomial_zero():
    torch.testing.assert_close(
        beignet.probabilists_hermite_polynomial_zero,
        torch.tensor([0.0]),
        check_dtype=False,
    )

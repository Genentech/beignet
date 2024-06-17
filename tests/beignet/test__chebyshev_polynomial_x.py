import beignet
import torch


def test_chebyshev_polynomial_x():
    torch.testing.assert_close(
        beignet.chebyshev_polynomial_x,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )

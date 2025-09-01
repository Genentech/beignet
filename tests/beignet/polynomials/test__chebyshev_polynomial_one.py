import torch

import beignet


def test_chebyshev_polynomial_one():
    torch.testing.assert_close(
        beignet.chebyshev_polynomial_one,
        torch.tensor([1.0]),
        check_dtype=False,
    )

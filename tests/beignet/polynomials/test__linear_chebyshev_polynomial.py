import torch

import beignet.polynomials


def test_linear_chebyshev_polynomial():
    torch.testing.assert_close(
        beignet.polynomials.linear_chebyshev_polynomial(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

import torch

import beignet.polynomials


def test_legendre_polynomial_x():
    torch.testing.assert_close(
        beignet.polynomials.legendre_polynomial_x,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )

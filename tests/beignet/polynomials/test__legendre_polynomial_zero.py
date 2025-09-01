import torch

import beignet.polynomials


def test_legendre_polynomial_zero():
    torch.testing.assert_close(
        beignet.polynomials.legendre_polynomial_zero,
        torch.tensor([0.0]),
        check_dtype=False,
    )

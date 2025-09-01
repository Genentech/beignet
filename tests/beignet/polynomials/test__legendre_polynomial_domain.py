import torch

import beignet.polynomials


def test_legendre_polynomial_domain():
    torch.testing.assert_close(
        beignet.polynomials.legendre_polynomial_domain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )

import torch

import beignet.polynomials


def test_laguerre_polynomial_domain():
    torch.testing.assert_close(
        beignet.polynomials.laguerre_polynomial_domain,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )

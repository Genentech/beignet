import torch

import beignet.polynomials


def test_laguerre_polynomial_one():
    torch.testing.assert_close(
        beignet.polynomials.laguerre_polynomial_one,
        torch.tensor([1.0]),
        check_dtype=False,
    )

import torch

import beignet.polynomials


def test_polynomial_one():
    torch.testing.assert_close(
        beignet.polynomials.polynomial_one,
        torch.tensor([1.0]),
        check_dtype=False,
    )

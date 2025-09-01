import torch

import beignet.polynomials


def test_physicists_hermite_polynomial_x():
    torch.testing.assert_close(
        beignet.polynomials.physicists_hermite_polynomial_x,
        torch.tensor([0, 0.5]),
        check_dtype=False,
    )

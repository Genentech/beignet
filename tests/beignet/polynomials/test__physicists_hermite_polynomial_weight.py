import torch

import beignet.polynomials


def test_physicists_hermite_polynomial_weight():
    torch.testing.assert_close(
        beignet.polynomials.physicists_hermite_polynomial_weight(
            torch.linspace(-5, 5, 11)
        ),
        torch.exp(-(torch.linspace(-5, 5, 11) ** 2)),
    )

import beignet
import torch


def test_physicists_hermite_polynomial_weight():
    torch.testing.assert_close(
        beignet.physicists_hermite_polynomial_weight(torch.linspace(-5, 5, 11)),
        torch.exp(-(torch.linspace(-5, 5, 11) ** 2)),
    )

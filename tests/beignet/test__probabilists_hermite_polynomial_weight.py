import beignet
import torch


def test_probabilists_hermite_polynomial_weight():
    torch.testing.assert_close(
        beignet.probabilists_hermite_polynomial_weight(
            torch.linspace(-5, 5, 11),
        ),
        torch.exp(-0.5 * torch.linspace(-5, 5, 11) ** 2),
    )

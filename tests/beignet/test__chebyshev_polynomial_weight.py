import beignet
import torch


def test_chebyshev_polynomial_weight():
    x = torch.linspace(-1, 1, 11)[1:-1]

    torch.testing.assert_close(
        beignet.chebyshev_polynomial_weight(x),
        1.0 / (torch.sqrt(1 + x) * torch.sqrt(1 - x)),
    )

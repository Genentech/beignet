import beignet.polynomial
import beignet.polynomial._chebweight
import torch


def test_chebweight():
    x = torch.linspace(-1, 1, 11)[1:-1]
    torch.testing.assert_close(
        beignet.polynomial.chebweight(x),
        1.0 / (torch.sqrt(1 + x) * torch.sqrt(1 - x)),
    )

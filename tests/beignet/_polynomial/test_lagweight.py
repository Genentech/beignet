import beignet.polynomial
import torch


def test_lagweight():
    x = torch.linspace(0, 10, 11)

    torch.testing.assert_close(
        beignet.polynomial.lagweight(x),
        torch.exp(-x),
    )

import beignet.polynomial
import torch


def test_legweight():
    torch.testing.assert_close(
        beignet.polynomial.legweight(
            torch.linspace(-1, 1, 11),
        ),
        torch.tensor([1.0]),
    )

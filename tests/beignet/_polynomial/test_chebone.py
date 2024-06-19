import beignet.polynomial
import torch


def test_chebone():
    torch.testing.assert_close(
        beignet.polynomial.chebone,
        torch.tensor([1]),
    )

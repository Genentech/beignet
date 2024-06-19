import beignet.polynomial
import torch


def test_legzero():
    torch.testing.assert_close(
        beignet.polynomial.legzero,
        torch.tensor([0]),
    )

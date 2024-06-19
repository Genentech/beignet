import beignet.polynomial
import torch


def test_polydomain():
    torch.testing.assert_close(
        beignet.polynomial.polydomain,
        torch.tensor([-1.0, 1.0]),
    )

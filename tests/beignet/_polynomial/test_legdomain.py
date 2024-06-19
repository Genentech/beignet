import beignet.polynomial
import torch


def test_legdomain():
    torch.testing.assert_close(
        beignet.polynomial.legdomain,
        torch.tensor([-1.0, 1.0]),
    )

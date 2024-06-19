import beignet.polynomial
import torch


def test_hermedomain():
    torch.testing.assert_close(
        beignet.polynomial.hermedomain,
        torch.tensor([-1.0, 1.0]),
    )

import beignet.polynomial
import beignet.polynomial._lagdomain
import torch


def test_lagdomain():
    torch.testing.assert_close(
        beignet.polynomial._lagdomain.lagdomain,
        torch.tensor([0.0, 1.0]),
    )

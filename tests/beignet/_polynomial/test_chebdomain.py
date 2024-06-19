import beignet.polynomial
import torch


def test_chebdomain():
    torch.testing.assert_close(
        beignet.polynomial.chebdomain,
        torch.tensor([-1, 1]),
    )

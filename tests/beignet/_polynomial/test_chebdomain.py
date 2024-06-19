import beignet.polynomial
import beignet.polynomial._chebdomain
import torch


def test_chebdomain():
    torch.testing.assert_close(
        beignet.polynomial._chebdomain.chebdomain,
        torch.tensor([-1.0, 1.0]),
    )

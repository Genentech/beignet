import beignet.polynomial
import beignet.polynomial._hermedomain
import torch


def test_hermedomain():
    torch.testing.assert_close(
        beignet.polynomial._hermedomain.hermedomain,
        torch.tensor([-1.0, 1.0]),
    )

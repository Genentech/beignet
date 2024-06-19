import beignet.polynomial
import beignet.polynomial._hermeone
import torch


def test_hermeone():
    torch.testing.assert_close(
        beignet.polynomial._hermeone.hermeone,
        torch.tensor([1]),
    )

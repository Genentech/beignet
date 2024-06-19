import beignet.polynomial
import beignet.polynomial._polyone
import torch


def test_polyone():
    torch.testing.assert_close(
        beignet.polynomial._polyone.polyone,
        torch.tensor([1]),
    )

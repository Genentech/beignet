import beignet.polynomial
import beignet.polynomial._hermzero
import torch


def test_hermzero():
    torch.testing.assert_close(
        beignet.polynomial._hermzero.hermzero,
        torch.tensor([0]),
    )

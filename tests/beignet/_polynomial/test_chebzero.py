import beignet.polynomial
import beignet.polynomial._chebzero
import torch


def test_chebzero():
    torch.testing.assert_close(
        beignet.polynomial._chebzero.chebzero,
        torch.tensor([0]),
    )

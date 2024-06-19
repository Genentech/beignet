import beignet.polynomial
import beignet.polynomial._hermezero
import torch


def test_hermezero():
    torch.testing.assert_close(
        beignet.polynomial._hermezero.hermezero,
        torch.tensor([0]),
    )

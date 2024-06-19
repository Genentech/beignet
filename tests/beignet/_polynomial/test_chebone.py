import beignet.polynomial
import beignet.polynomial._chebone
import torch


def test_chebone():
    torch.testing.assert_close(
        beignet.polynomial._chebone.chebone,
        torch.tensor([1]),
    )

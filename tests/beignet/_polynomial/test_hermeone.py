import beignet.polynomial
import beignet.polynomial._probabilists_hermite_series_one
import torch


def test_hermeone():
    torch.testing.assert_close(
        beignet.polynomial._hermeone.probabilists_hermite_series_one,
        torch.tensor([1]),
    )

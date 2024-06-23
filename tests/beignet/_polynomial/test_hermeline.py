import beignet.polynomial
import beignet.polynomial._probabilists_hermite_series_line
import torch


def test_hermeline():
    torch.testing.assert_close(
        beignet.polynomial._hermeline.probabilists_hermite_series_line(3, 4),
        torch.tensor([3, 4]),
    )

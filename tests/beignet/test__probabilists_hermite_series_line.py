import beignet.polynomial
import torch


def test_probabilists_hermite_series_line():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_line(3, 4),
        torch.tensor([3, 4]),
    )

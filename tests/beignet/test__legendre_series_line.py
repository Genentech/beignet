import beignet.polynomial
import beignet.polynomial._legendre_series_line
import torch


def test_legendre_series_line():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_line(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial.legendre_series_line(3, 0),
        torch.tensor([3]),
    )

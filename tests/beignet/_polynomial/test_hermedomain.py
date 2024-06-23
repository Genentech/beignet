import beignet.polynomial
import torch


def test_hermedomain():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_domain,
        torch.tensor([-1.0, 1.0]),
    )

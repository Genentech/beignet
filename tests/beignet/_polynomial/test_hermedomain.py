import beignet.polynomial
import beignet.polynomial._probabilists_hermite_series_domain
import torch


def test_hermedomain():
    torch.testing.assert_close(
        beignet.polynomial._hermedomain.probabilists_hermite_series_domain,
        torch.tensor([-1.0, 1.0]),
    )

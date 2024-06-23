import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_domain
import torch


def test_hermdomain():
    torch.testing.assert_close(
        beignet.polynomial._hermdomain.physicists_hermite_series_domain,
        torch.tensor([-1.0, 1.0]),
    )

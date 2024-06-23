import beignet.polynomial
import beignet.polynomial._power_series_domain
import torch


def test_polydomain():
    torch.testing.assert_close(
        beignet.polynomial._polydomain.power_series_domain,
        torch.tensor([-1.0, 1.0]),
    )

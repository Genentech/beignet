import beignet.polynomial
import beignet.polynomial._chebyshev_series_domain
import torch


def test_chebdomain():
    torch.testing.assert_close(
        beignet.polynomial._chebdomain.chebyshev_series_domain,
        torch.tensor([-1.0, 1.0]),
    )

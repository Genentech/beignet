import beignet.polynomial
import beignet.polynomial._polydomain
import torch


def test_polydomain():
    torch.testing.assert_close(
        beignet.polynomial._polydomain.polydomain,
        torch.tensor([-1.0, 1.0]),
    )

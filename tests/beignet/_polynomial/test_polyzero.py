import beignet.polynomial
import beignet.polynomial._polyzero
import torch.testing


def test_polyzero():
    torch.testing.assert_close(
        beignet.polynomial._polyzero.polyzero,
        torch.tensor([0]),
    )

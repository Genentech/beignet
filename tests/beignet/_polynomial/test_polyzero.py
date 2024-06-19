import beignet.polynomial
import torch.testing


def test_polyzero():
    torch.testing.assert_close(
        beignet.polynomial.polyzero,
        torch.tensor([0]),
    )

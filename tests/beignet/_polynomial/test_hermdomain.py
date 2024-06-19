import beignet.polynomial
import beignet.polynomial._hermdomain
import torch


def test_hermdomain():
    torch.testing.assert_close(
        beignet.polynomial._hermdomain.hermdomain,
        torch.tensor([-1.0, 1.0]),
    )

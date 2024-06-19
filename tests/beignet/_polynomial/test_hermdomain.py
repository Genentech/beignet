import beignet.polynomial
import torch


def test_hermdomain():
    torch.testing.assert_close(
        beignet.polynomial.hermdomain,
        torch.tensor([-1.0, 1.0]),
    )

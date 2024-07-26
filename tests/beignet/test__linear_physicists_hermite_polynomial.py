import beignet
import torch


def test_linear_physicists_hermite_polynomial():
    torch.testing.assert_close(
        beignet.linear_physicists_hermite_polynomial(3, 4),
        torch.tensor([3.0, 2.0]),
    )

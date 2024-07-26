import beignet
import torch


def test_physicists_hermite_polynomial_x():
    torch.testing.assert_close(
        beignet.physicists_hermite_polynomial_x,
        torch.tensor([0, 0.5]),
        check_dtype=False,
    )

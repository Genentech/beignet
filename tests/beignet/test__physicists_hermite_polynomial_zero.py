import torch

import beignet


def test_physicists_hermite_polynomial_zero():
    torch.testing.assert_close(
        beignet.physicists_hermite_polynomial_zero,
        torch.tensor([0.0]),
        check_dtype=False,
    )

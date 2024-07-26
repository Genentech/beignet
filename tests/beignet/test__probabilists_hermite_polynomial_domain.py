import beignet
import torch


def test_probabilists_hermite_polynomial_domain():
    torch.testing.assert_close(
        beignet.probabilists_hermite_polynomial_domain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )

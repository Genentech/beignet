import pytest
import torch

import beignet


def test_trim_physicists_hermite_polynomial_coefficients():
    with pytest.raises(ValueError):
        beignet.trim_physicists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        beignet.trim_physicists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        beignet.trim_physicists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        beignet.trim_physicists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )

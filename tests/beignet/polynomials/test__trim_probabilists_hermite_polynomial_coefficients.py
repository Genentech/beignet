import pytest
import torch

import beignet.polynomials


def test_trim_probabilists_hermite_polynomial_coefficients():
    with pytest.raises(ValueError):
        beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            -1,
        )

    torch.testing.assert_close(
        beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            2,
        ),
        torch.tensor([0.0]),
    )

import pytest
import torch

import beignet.polynomials


def test_physicists_hermite_polynomial_companion():
    with pytest.raises(ValueError):
        beignet.polynomials.physicists_hermite_polynomial_companion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.polynomials.physicists_hermite_polynomial_companion(
            torch.tensor([1.0]),
        )

    for index in range(1, 5):
        output = beignet.polynomials.physicists_hermite_polynomial_companion(
            torch.tensor([0.0] * index + [1.0]),
        )

        assert output.shape == (index, index)

    output = beignet.polynomials.physicists_hermite_polynomial_companion(
        torch.tensor([1.0, 2.0]),
    )

    assert output[0, 0] == -0.25

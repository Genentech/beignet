import pytest
import torch

import beignet


def test_evaluate_physicists_hermite_polynomial_3d(float64):
    input = torch.rand(3, 5) * 2 - 1

    a, b, c = input

    x, y, z = beignet.evaluate_polynomial(input, torch.tensor([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        beignet.evaluate_physicists_hermite_polynomial_3d(
            a,
            b,
            c[:2],
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([2.5, 1.0, 0.75]),
                torch.tensor([2.5, 1.0, 0.75]),
                torch.tensor([2.5, 1.0, 0.75]),
            ),
        )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial_3d(
            a,
            b,
            c,
            torch.einsum(
                "i,j,k->ijk",
                torch.tensor([2.5, 1.0, 0.75]),
                torch.tensor([2.5, 1.0, 0.75]),
                torch.tensor([2.5, 1.0, 0.75]),
            ),
        ),
        x * y * z,
    )

    output = beignet.evaluate_physicists_hermite_polynomial_3d(
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.ones([2, 3]),
        torch.einsum(
            "i,j,k->ijk",
            torch.tensor([2.5, 1.0, 0.75]),
            torch.tensor([2.5, 1.0, 0.75]),
            torch.tensor([2.5, 1.0, 0.75]),
        ),
    )

    assert output.shape == (2, 3)

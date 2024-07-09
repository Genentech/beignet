import beignet
import torch


def test_physicists_hermite_polynomial_vandermonde_3d():
    a, b, x3 = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = beignet.physicists_hermite_polynomial_vandermonde_3d(
        a,
        b,
        x3,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_physicists_hermite_polynomial_3d(a, b, x3, coefficients),
    )

    output = beignet.physicists_hermite_polynomial_vandermonde_3d(
        a,
        b,
        x3,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)

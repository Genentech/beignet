import beignet
import torch


def test_evaluate_legendre_polynomial_3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    target = beignet.evaluate_legendre_polynomial_3d(
        a,
        b,
        c,
        coefficients,
    )

    output = beignet.legendre_polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        target,
    )

    output = beignet.legendre_polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)

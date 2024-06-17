import beignet
import torch


def test_legendre_polynomial_vandermonde_2d():
    a, b, x3 = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = beignet.legendre_polynomial_vandermonde_2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )
    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_legendre_polynomial_2d(
            a,
            b,
            coefficients,
        ),
    )

    output = beignet.legendre_polynomial_vandermonde_2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)

import beignet
import torch


def test_chebyshev_polynomial_vandermonde_2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = beignet.chebyshev_polynomial_vandermonde_2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_chebyshev_polynomial_2d(
            a,
            b,
            coefficients,
        ),
    )

    van = beignet.chebyshev_polynomial_vandermonde_2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert van.shape == (5, 6)

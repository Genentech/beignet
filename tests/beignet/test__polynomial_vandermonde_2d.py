import torch

import beignet


def test_polynomial_vandermonde_2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = beignet.polynomial_vandermonde_2d(a, b, degree=torch.tensor([1, 2]))

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_polynomial_2d(
            a,
            b,
            coefficients,
        ),
    )

    output = beignet.polynomial_vandermonde_2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)

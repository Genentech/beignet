import torch

import beignet


def test_probabilists_hermite_polynomial_vandermonde_3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = beignet.probabilists_hermite_polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_probabilists_hermite_polynomial_3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = beignet.probabilists_hermite_polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1, 2, 3]),
    )

    assert output.shape == (5, 24)

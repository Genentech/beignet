import beignet
import torch


def test_polynomial_vandermonde_3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = beignet.polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        beignet.evaluate_polynomial_3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = beignet.polynomial_vandermonde_3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert output.shape == (5, 24)

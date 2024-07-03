import beignet
import torch


def test_probabilists_hermite_polynomial_vandermonde():
    x = torch.arange(3)
    v = beignet.probabilists_hermite_polynomial_vandermonde(
        x,
        3,
    )
    assert v.shape == (3, 4)
    for i in range(4):
        coefficients = torch.tensor([0.0] * i + [1.0])
        torch.testing.assert_close(
            v[..., i],
            beignet.evaluate_probabilists_hermite_polynomial(x, coefficients),
        )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v = beignet.probabilists_hermite_polynomial_vandermonde(
        x,
        3,
    )
    assert v.shape == (3, 2, 4)
    for i in range(4):
        coefficients = torch.tensor([0.0] * i + [1.0])
        torch.testing.assert_close(
            v[..., i],
            beignet.evaluate_probabilists_hermite_polynomial(x, coefficients),
        )

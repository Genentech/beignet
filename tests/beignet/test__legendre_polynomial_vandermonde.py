import beignet
import pytest
import torch


def test_legendre_polynomial_vandermonde():
    x = torch.arange(3)

    v = beignet.legendre_polynomial_vandermonde(
        x,
        degree=3,
    )

    assert v.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.evaluate_legendre_polynomial(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    v = beignet.legendre_polynomial_vandermonde(
        x,
        degree=3,
    )

    assert v.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.evaluate_legendre_polynomial(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        beignet.legendre_polynomial_vandermonde(
            torch.tensor([1, 2, 3]),
            -1,
        )

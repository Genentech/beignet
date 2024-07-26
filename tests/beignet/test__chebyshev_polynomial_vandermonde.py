import beignet
import torch


def test_chebyshev_polynomial_vandermonde():
    v = beignet.chebyshev_polynomial_vandermonde(
        torch.arange(3),
        degree=torch.tensor([3]),
    )

    assert v.shape == (3, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            beignet.evaluate_chebyshev_polynomial(
                torch.arange(3),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    v = beignet.chebyshev_polynomial_vandermonde(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=torch.tensor([3]),
    )

    assert v.shape == (3, 2, 4)

    for i in range(4):
        torch.testing.assert_close(
            v[..., i],
            beignet.evaluate_chebyshev_polynomial(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

import beignet
import pytest
import torch


def test_polynomial_vandermonde():
    output = beignet.polynomial_vandermonde(
        torch.arange(3.0),
        degree=torch.tensor([3]),
    )

    assert output.shape == (3, 4)

    for i in range(4):
        torch.testing.assert_close(
            output[..., i],
            beignet.evaluate_polynomial(
                torch.arange(3),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    output = beignet.polynomial_vandermonde(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=torch.tensor([3]),
    )

    assert output.shape == (3, 2, 4)

    for i in range(4):
        torch.testing.assert_close(
            output[..., i],
            beignet.evaluate_polynomial(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * i + [1.0]),
            ),
        )

    with pytest.raises(ValueError):
        beignet.polynomial_vandermonde(
            torch.arange(3),
            degree=torch.tensor([-1]),
        )

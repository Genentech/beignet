import beignet
import torch


def test_physicists_hermite_polynomial_vandermonde():
    x = torch.arange(3)

    output = beignet.physicists_hermite_polynomial_vandermonde(
        x,
        degree=3,
    )

    assert output.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.evaluate_physicists_hermite_polynomial(
                x,
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

    output = beignet.physicists_hermite_polynomial_vandermonde(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        degree=3,
    )

    assert output.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.evaluate_physicists_hermite_polynomial(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([0.0] * index + [1.0]),
            ),
        )

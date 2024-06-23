import beignet.polynomial
import torch.testing


def test_physicists_hermite_series_vandermonde_1d():
    v = beignet.polynomial.physicists_hermite_series_vandermonde_1d(
        torch.arange(3),
        3,
    )

    assert v.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_physicists_hermite_series_1d(
                torch.arange(3),
                torch.tensor([0] * index + [1]),
            ),
        )

    v = beignet.polynomial.physicists_hermite_series_vandermonde_1d(
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        3,
    )

    assert v.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_physicists_hermite_series_1d(
                torch.tensor([[1, 2], [3, 4], [5, 6]]),
                torch.tensor([0] * index + [1]),
            ),
        )

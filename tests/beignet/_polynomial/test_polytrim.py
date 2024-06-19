import beignet.polynomial
import torch


def test_polytrim():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    torch.testing.assert_allclose(
        beignet.polynomial.polytrim(coef),
        coef[:-1],
    )

    torch.testing.assert_allclose(
        beignet.polynomial.polytrim(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_allclose(
        beignet.polynomial.polytrim(coef, 2),
        torch.tensor([0], dtype=torch.float64),
    )

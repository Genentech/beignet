import beignet.polynomial
import beignet.polynomial._chebtrim
import torch


def test_chebtrim():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    torch.testing.assert_close(
        beignet.polynomial._chebtrim.chebtrim(coef),
        coef[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomial._chebtrim.chebtrim(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial._chebtrim.chebtrim(coef, 2),
        torch.tensor([0], dtype=torch.float64),
    )

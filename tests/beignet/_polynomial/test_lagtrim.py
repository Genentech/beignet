import beignet.polynomial
import beignet.polynomial._lagtrim
import torch


def test_lagtrim():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    torch.testing.assert_close(
        beignet.polynomial._lagtrim.lagtrim(coef),
        coef[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomial._lagtrim.lagtrim(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial._lagtrim.lagtrim(coef, 2),
        torch.tensor([0], dtype=torch.float64),
    )

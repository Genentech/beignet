import beignet.polynomial
import beignet.polynomial._hermtrim
import torch


def test_hermtrim():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    torch.testing.assert_close(
        beignet.polynomial._hermtrim.hermtrim(coef),
        coef[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomial._hermtrim.hermtrim(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial._hermtrim.hermtrim(coef, 2),
        torch.tensor([0], dtype=torch.float64),
    )

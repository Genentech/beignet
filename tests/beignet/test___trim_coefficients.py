import beignet.polynomial
import beignet.polynomial.__trim_coefficients
import torch


def test__trim_coefficients():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    torch.testing.assert_close(
        beignet.polynomial._trim_coefficients(
            coef,
        ),
        coef[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomial._trim_coefficients(
            coef,
            1,
        ),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial._trim_coefficients(
            coef,
            2,
        ),
        torch.tensor([0], dtype=torch.float64),
    )

import beignet.polynomial
import beignet.polynomial._legmulx
import torch


def test_legmulx():
    torch.testing.assert_close(
        beignet.polynomial._legmulx.legmulx(
            torch.tensor([0]),
        ),
        torch.tensor([0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial._legmulx.legmulx(
            torch.tensor([1]),
        ),
        torch.tensor([0, 1], dtype=torch.float64),
    )

    for i in range(1, 5):
        tmp = 2 * i + 1

        torch.testing.assert_close(
            beignet.polynomial._legmulx.legmulx(
                torch.tensor([0] * i + [1]),
            ),
            torch.tensor(
                [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp], dtype=torch.float64
            ),
        )

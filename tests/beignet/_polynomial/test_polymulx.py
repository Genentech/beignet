import beignet.polynomial
import beignet.polynomial._polymulx
import torch


def test_polymulx():
    torch.testing.assert_close(
        beignet.polynomial._polymulx.polymulx(
            torch.tensor([0]),
        ),
        torch.tensor([0.0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial._polymulx.polymulx(
            torch.tensor([1]),
        ),
        torch.tensor([0, 1], dtype=torch.float64),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial._polymulx.polymulx(
                torch.tensor([0] * index + [1]),
            ),
            torch.tensor([0] * (index + 1) + [1], dtype=torch.float64),
        )

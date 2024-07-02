import torch
from beignet.polynomial import polymulx


def test_polymulx():
    torch.testing.assert_close(
        polymulx(
            torch.tensor([0.0]),
        ),
        torch.tensor([0.0, 0.0]),
    )

    torch.testing.assert_close(
        polymulx(
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0, 1.0]),
    )

    for i in range(1, 5):
        torch.testing.assert_close(
            polymulx(
                torch.tensor([0.0] * i + [1.0]),
            ),
            torch.tensor([0.0] * (i + 1) + [1.0]),
        )

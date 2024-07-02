import pytest
import torch
from beignet.polynomial import polytrim


def test_polytrim():
    with pytest.raises(ValueError):
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=-1,
        )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-1],
    )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=1,
        ),
        torch.tensor([2.0, -1.0, 1.0, 0.0])[:-3],
    )

    torch.testing.assert_close(
        polytrim(
            torch.tensor([2.0, -1.0, 1.0, 0.0]),
            tol=2,
        ),
        torch.tensor([0.0]),
    )

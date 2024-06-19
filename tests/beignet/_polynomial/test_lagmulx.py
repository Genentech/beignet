import beignet.polynomial
import torch


def test_lagmulx():
    torch.testing.assert_close(
        beignet.polynomial.lagmulx(
            torch.tensor([0]),
        ),
        torch.tensor([0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial.lagmulx(
            torch.tensor([1]),
        ),
        torch.tensor([1, -1], dtype=torch.float64),
    )

    for index in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial.lagmulx(
                torch.tensor(
                    [0] * index + [1],
                ),
            ),
            torch.tensor(
                [0] * (index - 1) + [-index, 2 * index + 1, -(index + 1)],
                dtype=torch.float64,
            ),
        )

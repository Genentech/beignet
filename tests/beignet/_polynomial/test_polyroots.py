import beignet.polynomial
import torch.testing


def test_polyroots():
    torch.testing.assert_close(
        beignet.polynomial.polyroots(
            torch.tensor([1]),
        ),
        torch.tensor([], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial.polyroots(
            torch.tensor([1, 2]),
        ),
        torch.tensor([-0.5], dtype=torch.float64),
    )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.polytrim(
                beignet.polynomial.polyroots(
                    beignet.polynomial.polyfromroots(
                        torch.linspace(-1, 1, i),
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.polytrim(
                torch.linspace(-1, 1, i),
                tolerance=1e-6,
            ),
        )

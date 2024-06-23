import beignet.polynomial
import torch.testing


def test_polyroots():
    torch.testing.assert_close(
        beignet.polynomial.power_series_roots(
            torch.tensor([1]),
        ),
        torch.tensor([], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial.power_series_roots(
            torch.tensor([1, 2]),
        ),
        torch.tensor([-0.5], dtype=torch.float64),
    )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(
                beignet.polynomial.power_series_roots(
                    beignet.polynomial.polyfromroots(
                        torch.linspace(-1, 1, i),
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_power_series(
                torch.linspace(-1, 1, i),
                tolerance=1e-6,
            ),
        )

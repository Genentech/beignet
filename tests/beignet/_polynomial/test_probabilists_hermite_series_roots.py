import beignet.polynomial
import torch.testing


def test_probabilists_hermite_series_roots():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_roots(
            torch.tensor([1]),
        ),
        torch.tensor([], dtype=torch.float64),
    )

    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_roots(
            torch.tensor([1, 1]),
        ),
        torch.tensor([-1], dtype=torch.float64),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.trim_probabilists_hermite_series(
                beignet.polynomial.probabilists_hermite_series_roots(
                    beignet.polynomial.probabilists_hermite_series_from_roots(
                        torch.linspace(-1, 1, index),
                    )
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_probabilists_hermite_series(
                torch.linspace(-1, 1, index),
                tolerance=1e-6,
            ),
        )

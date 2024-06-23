import beignet.polynomial
import torch.testing

from tests.beignet._polynomial.test_polynomial import polynomial_coefficients


def test_power_series_from_roots():
    torch.testing.assert_close(
        beignet.polynomial.trim_power_series(
            beignet.polynomial.power_series_from_roots(
                torch.tensor([]),
            ),
            tolerance=0.000001,
        ),
        torch.tensor([1], dtype=torch.float32),
    )

    for i in range(1, 5):
        roots = torch.cos(torch.linspace(-torch.pi, 0, 2 * i + 1)[1::2])

        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(
                beignet.polynomial.power_series_from_roots(
                    roots,
                )
                * 2 ** (i - 1),
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_power_series(
                polynomial_coefficients[i],
                tolerance=0.000001,
            ),
        )

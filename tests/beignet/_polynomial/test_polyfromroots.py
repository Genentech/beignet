import beignet.polynomial
import beignet.polynomial._polyfromroots
import beignet.polynomial._polytrim
import torch.testing

from tests.beignet._polynomial.test_polynomial import polynomial_coefficients


def test_polyfromroots():
    torch.testing.assert_close(
        beignet.polynomial.polytrim(
            beignet.polynomial.polyfromroots(
                torch.tensor([]),
            ),
            tolerance=1e-6,
        ),
        torch.tensor([1], dtype=torch.float32),
    )

    for i in range(1, 5):
        roots = torch.cos(torch.linspace(-torch.pi, 0, 2 * i + 1)[1::2])

        torch.testing.assert_close(
            beignet.polynomial.polytrim(
                beignet.polynomial.polyfromroots(
                    roots,
                )
                * 2 ** (i - 1),
                tolerance=1e-6,
            ),
            beignet.polynomial.polytrim(
                polynomial_coefficients[i],
                tolerance=1e-6,
            ),
        )

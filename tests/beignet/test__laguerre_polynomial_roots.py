import beignet
import torch


def test_laguerre_polynomial_roots():
    torch.testing.assert_close(
        beignet.laguerre_polynomial_roots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        beignet.laguerre_polynomial_roots(
            torch.tensor([0.0, 1.0]),
        ),
        torch.tensor([1.0]),
    )

    for index in range(2, 5):
        torch.testing.assert_close(
            beignet.trim_laguerre_polynomial_coefficients(
                beignet.laguerre_polynomial_roots(
                    beignet.laguerre_polynomial_from_roots(
                        torch.linspace(0, 3, index),
                    ),
                ),
                tol=0.000001,
            ),
            beignet.trim_laguerre_polynomial_coefficients(
                torch.linspace(0, 3, index),
                tol=0.000001,
            ),
        )

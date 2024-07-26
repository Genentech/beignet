import beignet
import torch


def test_physicists_hermite_polynomial_roots():
    torch.testing.assert_close(
        beignet.physicists_hermite_polynomial_roots(
            torch.tensor([1.0]),
        ),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        beignet.physicists_hermite_polynomial_roots(
            torch.tensor([1.0, 1.0]),
        ),
        torch.tensor([-0.5]),
    )

    for i in range(2, 5):
        input = torch.linspace(-1, 1, i)

        torch.testing.assert_close(
            beignet.trim_physicists_hermite_polynomial_coefficients(
                beignet.physicists_hermite_polynomial_roots(
                    beignet.physicists_hermite_polynomial_from_roots(
                        input,
                    ),
                ),
                tol=0.000001,
            ),
            beignet.trim_physicists_hermite_polynomial_coefficients(
                input,
                tol=0.000001,
            ),
        )

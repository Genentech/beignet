import math

import torch

import beignet.polynomials


def test_probabilists_hermite_polynomial_from_roots():
    torch.testing.assert_close(
        beignet.polynomials.trim_probabilists_hermite_polynomial_coefficients(
            beignet.polynomials.probabilists_hermite_polynomial_from_roots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for i in range(1, 5):
        roots = torch.cos(torch.linspace(-math.pi, 0, 2 * i + 1)[1::2])

        pol = beignet.polynomials.probabilists_hermite_polynomial_from_roots(roots)

        assert len(pol) == i + 1

        torch.testing.assert_close(
            beignet.polynomials.probabilists_hermite_polynomial_to_polynomial(pol)[-1],
            torch.tensor(1.0),
        )

        # torch.testing.assert_close(
        #     beignet.polynomials.hermeval(roots, pol),
        #     torch.tensor([0.0]),
        # )

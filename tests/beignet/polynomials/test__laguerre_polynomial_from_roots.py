import torch

import beignet.polynomials


def test_laguerre_polynomial_from_roots():
    torch.testing.assert_close(
        beignet.polynomials.trim_laguerre_polynomial_coefficients(
            beignet.polynomials.laguerre_polynomial_from_roots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    # for i in range(1, 5):
    #     roots = torch.linspace(-math.pi, 0, 2 * i + 1)
    #
    #     roots = roots[1::2]
    #
    #     roots = torch.cos(roots)
    #
    #     output = beignet.polynomials.lag2poly(
    #         beignet.polynomials.lagfromroots(
    #             roots,
    #         ),
    #     )
    #
    #     torch.testing.assert_close(
    #         output,
    #         torch.tensor([1.0]),
    #     )
    #
    #     output = beignet.polynomials.lagval(
    #         roots,
    #         beignet.polynomials.lagfromroots(
    #             roots,
    #         ),
    #     )
    #
    #     torch.testing.assert_close(
    #         output,
    #         torch.tensor([0.0]),
    #     )

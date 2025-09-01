import torch

import beignet.polynomials


def test_physicists_hermite_polynomial_from_roots():
    torch.testing.assert_close(
        beignet.polynomials.trim_physicists_hermite_polynomial_coefficients(
            beignet.polynomials.physicists_hermite_polynomial_from_roots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    # for i in range(1, 5):
    #     roots = torch.cos(torch.linspace(-math.pi, 0, 2 * i + 1)[1::2])
    #     target = 0
    #
    #     torch.testing.assert_close(
    #         beignet.polynomials.herm2poly(
    #             beignet.polynomials.hermfromroots(
    #                 roots,
    #             ),
    #         )[-1],
    #         torch.tensor([1.0]),
    #     )
    #
    #     torch.testing.assert_close(
    #         beignet.polynomials.hermval(
    #             roots,
    #             beignet.polynomials.hermfromroots(
    #                 roots,
    #             ),
    #         ),
    #         target,
    #     )

import beignet
import torch


def test_laguerre_polynomial_from_roots():
    torch.testing.assert_close(
        beignet.trim_laguerre_polynomial_coefficients(
            beignet.laguerre_polynomial_from_roots(
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
    #     output = beignet.lag2poly(
    #         beignet.lagfromroots(
    #             roots,
    #         ),
    #     )
    #
    #     torch.testing.assert_close(
    #         output,
    #         torch.tensor([1.0]),
    #     )
    #
    #     output = beignet.lagval(
    #         roots,
    #         beignet.lagfromroots(
    #             roots,
    #         ),
    #     )
    #
    #     torch.testing.assert_close(
    #         output,
    #         torch.tensor([0.0]),
    #     )

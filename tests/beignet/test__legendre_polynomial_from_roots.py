import math

import beignet
import torch


def test_legendre_polynomial_from_roots():
    torch.testing.assert_close(
        beignet.trim_legendre_polynomial_coefficients(
            beignet.legendre_polynomial_from_roots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        input = torch.linspace(-math.pi, 0, 2 * index + 1)[1::2]

        output = beignet.legendre_polynomial_from_roots(
            torch.cos(
                input,
            ),
        )

        assert output.shape[-1] == index + 1

        # torch.testing.assert_close(
        #     beignet.leg2poly(
        #         beignet.legfromroots(
        #             torch.cos(
        #                 input,
        #             ),
        #         )
        #     )[-1],
        #     torch.tensor([1.0]),
        # )

        # torch.testing.assert_close(
        #     beignet.legval(
        #         torch.cos(
        #             input,
        #         ),
        #         beignet.legfromroots(
        #             torch.cos(
        #                 input,
        #             ),
        #         ),
        #     ),
        #     torch.tensor([0.0]),
        # )

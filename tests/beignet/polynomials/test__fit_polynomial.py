import torch
from torch import Tensor

import beignet.polynomials


def test_fit_polynomial(float64):
    def f(x: Tensor) -> Tensor:
        return x * (x - 1) * (x - 2)

    def g(x: Tensor) -> Tensor:
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.polynomials.evaluate_polynomial(
            input,
            beignet.polynomials.fit_polynomial(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomials.evaluate_polynomial(
            input,
            beignet.polynomials.fit_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomials.evaluate_polynomial(
            input,
            beignet.polynomials.fit_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomials.evaluate_polynomial(
            input,
            beignet.polynomials.fit_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomials.fit_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                beignet.polynomials.fit_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomials.fit_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomials.fit_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                beignet.polynomials.fit_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomials.fit_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         input,
    #         other.at[0::2].set(0),
    #         degree=3,
    #         weight=weight,
    #     ),
    #     beignet.polynomials.polyfit(
    #         input,
    #         other,
    #         degree=torch.tensor([0, 1, 2, 3]),
    #     ),
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         input,
    #         other.at[0::2].set(0),
    #         degree=torch.tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     beignet.polynomials.polyfit(
    #         input,
    #         other,
    #         degree=torch.tensor([0, 1, 2, 3]),
    #     ),
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         input,
    #         torch.tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
    #         degree=3,
    #         weight=weight,
    #     ),
    #     torch.tensor(
    #         [
    #             beignet.polynomials.polyfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #             beignet.polynomials.polyfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         input,
    #         torch.tensor([other.at[0::2].set(0), other.at[0::2].set(0)]).T,
    #         degree=torch.tensor([0, 1, 2, 3]),
    #         weight=weight,
    #     ),
    #     torch.tensor(
    #         [
    #             beignet.polynomials.polyfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #             beignet.polynomials.polyfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         1,
    #     ),
    #     torch.tensor([0, 1]),
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         torch.tensor([1, 1j, -1, -0 - 1j]),
    #         torch.tensor([1, 1j, -1, -0 - 1j]),
    #         (0, 1),
    #     ),
    #     torch.tensor([0, 1]),
    # )
    #
    # input = torch.linspace(-1, 1, 50)
    #
    # other = g(input)
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyval(
    #         input,
    #         beignet.polynomials.polyfit(
    #             input,
    #             other,
    #             degree=torch.tensor([4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyval(
    #         input,
    #         beignet.polynomials.polyfit(
    #             input,
    #             other,
    #             degree=torch.tensor([0, 2, 4]),
    #         ),
    #     ),
    #     other,
    # )
    #
    # torch.testing.assert_close(
    #     beignet.polynomials.polyfit(
    #         input,
    #         other,
    #         degree=torch.tensor([4]),
    #     ),
    #     beignet.polynomials.polyfit(
    #         input,
    #         other,
    #         degree=torch.tensor([0, 2, 4]),
    #     ),
    # )

import beignet
import torch


def test_fit_chebyshev_polynomial():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    # torch.testing.assert_close(
    #     beignet.chebfit(
    #         input,
    #         torch.stack([other, other]).T,
    #         degree=4,
    #     ),
    #     torch.stack(
    #         [
    #             beignet.chebfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #             beignet.chebfit(
    #                 input,
    #                 other,
    #                 degree=torch.tensor([0, 1, 2, 3]),
    #             ),
    #         ]
    #     ).T,
    # )

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_chebyshev_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    # torch.testing.assert_close(
    #     beignet.chebfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=torch.tensor([1]),
    #     ),
    #     torch.tensor([0, 1]),
    # )

    # torch.testing.assert_close(
    #     beignet.chebfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=torch.tensor([0, 1]),
    #     ),
    #     torch.tensor([0, 1]),
    # )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_chebyshev_polynomial(
            input,
            beignet.fit_chebyshev_polynomial(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=4,
        ),
        beignet.fit_chebyshev_polynomial(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )

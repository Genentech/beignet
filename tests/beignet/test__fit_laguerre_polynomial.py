import beignet
import torch


def test_fit_laguerre_polynomial():
    def f(x):
        return x * (x - 1) * (x - 2)

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial(
            input,
            beignet.fit_laguerre_polynomial(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial(
            input,
            beignet.fit_laguerre_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial(
            input,
            beignet.fit_laguerre_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_laguerre_polynomial(
            input,
            beignet.fit_laguerre_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.fit_laguerre_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.fit_laguerre_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_laguerre_polynomial(
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
        beignet.fit_laguerre_polynomial(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        beignet.fit_laguerre_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_laguerre_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        beignet.fit_laguerre_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_laguerre_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        beignet.fit_laguerre_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.fit_laguerre_polynomial(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    # torch.testing.assert_close(
    #     beignet.lagfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=torch.tensor([1]),
    #     ),
    #     torch.tensor([1, -1]),
    # )

    # torch.testing.assert_close(
    #     beignet.lagfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=torch.tensor([0, 1]),
    #     ),
    #     torch.tensor([1, -1]),
    # )

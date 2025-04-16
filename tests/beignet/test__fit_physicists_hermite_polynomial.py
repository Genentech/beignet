import torch

import beignet


def test_fit_physicists_hermite_polynomial(float64):
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    weight = torch.zeros_like(input)

    weight[1::2] = 1.0

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.fit_physicists_hermite_polynomial(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    # torch.testing.assert_close(
    #     beignet.hermfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=1,
    #     ),
    #     torch.tensor([0.0j, 0.5j]),
    # )

    # torch.testing.assert_close(
    #     beignet.hermfit(
    #         torch.tensor([1, 1j, -1, -1j]),
    #         torch.tensor([1, 1j, -1, -1j]),
    #         degree=torch.tensor([0, 1]),
    #     ),
    #     torch.tensor([0, 0.5]),
    # )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.evaluate_physicists_hermite_polynomial(
            input,
            beignet.fit_physicists_hermite_polynomial(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=4,
        ),
        beignet.fit_physicists_hermite_polynomial(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )

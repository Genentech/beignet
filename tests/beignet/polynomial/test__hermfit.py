import beignet.polynomial
import torch


def test_hermfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
        ),
        torch.stack(
            [
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.polynomial.hermfit(
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
        beignet.polynomial.hermfit(
            input,
            other,
            degree=3,
            weight=weight,
        ),
        beignet.polynomial.hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        beignet.polynomial.hermfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            torch.stack([other, other]).T,
            degree=3,
            weight=weight,
        ),
        torch.stack(
            [
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.stack(
            [
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
                (
                    beignet.polynomial.hermfit(
                        input,
                        other,
                        degree=torch.tensor([0, 1, 2, 3]),
                    )
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            torch.tensor([1, 1j, -1, -1j]),
            torch.tensor([1, 1j, -1, -1j]),
            degree=1,
        ),
        torch.tensor([0.0j, 0.5j]),
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            torch.tensor([1, 1j, -1, -1j]),
            torch.tensor([1, 1j, -1, -1j]),
            degree=torch.tensor([0, 1]),
        ),
        torch.tensor([0, 0.5]),
    )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermval(
            input,
            beignet.polynomial.hermfit(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.hermfit(
            input,
            other,
            degree=4,
        ),
        beignet.polynomial.hermfit(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )

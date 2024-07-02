import beignet.polynomial
import torch


def test_chebfit():
    def f(x):
        return x * (x - 1) * (x - 2)

    def g(x):
        return x**4 + x**2 + 1

    input = torch.linspace(0, 2, 50)

    other = f(input)

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=3,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=4,
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=torch.tensor([0, 1, 2, 3, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=torch.tensor([2, 3, 4, 1, 0]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            torch.stack([other, other]).T,
            degree=4,
        ),
        torch.stack(
            [
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            torch.stack([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
        torch.stack(
            [
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomial.chebfit(
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
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([3]),
            weight=weight,
        ),
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([0, 1, 2, 3]),
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            torch.tensor([other, other]).T,
            degree=torch.tensor([3]),
            weight=weight,
        ),
        torch.tensor(
            [
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            torch.tensor([other, other]).T,
            degree=torch.tensor([0, 1, 2, 3]),
            weight=weight,
        ),
        torch.tensor(
            [
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
                beignet.polynomial.chebfit(
                    input,
                    other,
                    degree=torch.tensor([0, 1, 2, 3]),
                ),
            ],
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            torch.tensor([1, 1j, -1, -1j]),
            torch.tensor([1, 1j, -1, -1j]),
            degree=torch.tensor([1]),
        ),
        torch.tensor([0, 1]),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            torch.tensor([1, 1j, -1, -1j]),
            torch.tensor([1, 1j, -1, -1j]),
            degree=torch.tensor([0, 1]),
        ),
        torch.tensor([0, 1]),
    )

    input = torch.linspace(-1, 1, 50)

    other = g(input)

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=torch.tensor([4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebval(
            input,
            beignet.polynomial.chebfit(
                input,
                other,
                degree=torch.tensor([0, 2, 4]),
            ),
        ),
        other,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([4]),
        ),
        beignet.polynomial.chebfit(
            input,
            other,
            degree=torch.tensor([0, 2, 4]),
        ),
    )

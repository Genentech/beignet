import beignet.polynomial
import beignet.polynomial._legval
import beignet.polynomial._multiply_legendre_series
import torch


def test_legmul():
    x = torch.linspace(-1, 1, 100)

    for j in range(5):
        val1 = beignet.polynomial._legval.legval(
            x,
            torch.tensor([0] * j + [1]),
        )

        for k in range(5):
            val2 = beignet.polynomial._legval.legval(
                x,
                torch.tensor([0] * k + [1]),
            )

            pol3 = beignet.polynomial._legmul.multiply_legendre_series(
                torch.tensor([0] * j + [1]),
                torch.tensor([0] * k + [1]),
            )

            val3 = beignet.polynomial._legval.legval(x, pol3)

            assert len(pol3) == j + k + 1

            torch.testing.assert_close(
                val3,
                val1 * val2,
            )

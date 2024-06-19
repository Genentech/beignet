import beignet.polynomial
import numpy
import torch.testing


def test_hermemul():
    x = numpy.linspace(-3, 3, 100)

    for j in range(5):
        val1 = beignet.polynomial.hermeval(
            x,
            torch.tensor([0] * j + [1]),
        )

        for k in range(5):
            val2 = beignet.polynomial.hermeval(
                x,
                torch.tensor([0] * k + [1]),
            )

            pol3 = beignet.polynomial.hermemul(
                torch.tensor([0] * j + [1]),
                torch.tensor([0] * k + [1]),
            )

            val3 = beignet.polynomial.hermeval(x, pol3)

            assert len(pol3) == j + k + 1

            torch.testing.assert_close(
                val3,
                val1 * val2,
            )

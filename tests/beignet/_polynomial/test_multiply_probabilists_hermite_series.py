import beignet.polynomial
import numpy
import torch.testing


def test_multiply_probabilists_hermite_series():
    x = numpy.linspace(-3, 3, 100)

    for j in range(5):
        val1 = beignet.polynomial.evaluate_1d_probabilists_hermite_series(
            x,
            torch.tensor([0] * j + [1]),
        )

        for k in range(5):
            val2 = beignet.polynomial.evaluate_1d_probabilists_hermite_series(
                x,
                torch.tensor([0] * k + [1]),
            )

            pol3 = beignet.polynomial.multiply_probabilists_hermite_series(
                torch.tensor([0] * j + [1]),
                torch.tensor([0] * k + [1]),
            )

            val3 = beignet.polynomial.evaluate_1d_probabilists_hermite_series(x, pol3)

            assert len(pol3) == j + k + 1

            torch.testing.assert_close(
                val3,
                val1 * val2,
            )

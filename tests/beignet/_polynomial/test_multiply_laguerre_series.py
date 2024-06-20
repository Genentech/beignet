import beignet.polynomial
import torch.testing


def test_multiply_laguerre_series():
    x = torch.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = torch.tensor([0] * i + [1])
        val1 = beignet.polynomial.evaluate_1d_laguerre_series(x, pol1)

        for j in range(5):
            pol2 = torch.tensor([0] * j + [1])
            val2 = beignet.polynomial.evaluate_1d_laguerre_series(x, pol2)
            pol3 = beignet.polynomial.multiply_laguerre_series(pol1, pol2)
            val3 = beignet.polynomial.evaluate_1d_laguerre_series(x, pol3)

            assert len(pol3) == i + j + 1

            torch.testing.assert_close(
                val3,
                val1 * val2,
                atol=1e-4,
                rtol=1e-4,
            )

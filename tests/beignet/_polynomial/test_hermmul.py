import beignet.polynomial
import beignet.polynomial._hermmul
import beignet.polynomial._hermval
import torch.testing


def test_hermmul():
    x = torch.linspace(-3, 3, 100)

    for j in range(5):
        pol1 = torch.tensor([0] * j + [1])
        val1 = beignet.polynomial.hermval(x, pol1)

        for k in range(5):
            pol2 = torch.tensor([0] * k + [1])
            val2 = beignet.polynomial.hermval(x, pol2)
            pol3 = beignet.polynomial.hermmul(pol1, pol2)
            val3 = beignet.polynomial.hermval(x, pol3)
            assert len(pol3) == j + k + 1
            torch.testing.assert_close(val3, val1 * val2, atol=1e-4, rtol=1e-4)

import beignet
import scipy.optimize
import torch.testing


def test_newton():
    def func(x):
        return x**3 - 1

    torch.testing.assert_close(
        beignet.newton(func, torch.tensor([1.5], dtype=torch.float64))[0],
        torch.tensor([scipy.optimize.newton(func, 1.5)], dtype=torch.float64),
    )

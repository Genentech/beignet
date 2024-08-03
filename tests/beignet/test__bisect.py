import torch
from beignet import bisect


def f(x, c):
    return x.pow(2) - c


# f(xstar(c), c) = 0
def xstar(c):
    return c.pow(0.5)


def test_bisect():
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0

    root = bisect(f, c, lower=lower, upper=upper, dtype=torch.float64)
    expected = xstar(c)

    torch.testing.assert_close(root, expected)


def test_bisect_grad():
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0

    grad = torch.func.vmap(
        torch.func.grad(
            lambda c: bisect(f, c, lower=lower, upper=upper, dtype=torch.float64)
        )
    )(c)

    expected = torch.func.vmap(torch.func.grad(xstar))(c)

    torch.testing.assert_close(grad, expected)

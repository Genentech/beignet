import beignet
import torch

# def test_root():
#    c = torch.linspace(2, 100, 1001, dtype=torch.float64)
#
#    output, _ = beignet.root(
#        lambda x: x**2 - c,
#        torch.sqrt(c) - 1.1,
#        torch.sqrt(c) + 1.0,
#    )
#
#    torch.testing.assert_close(output, torch.sqrt(c))


def f(x, c):
    return x.pow(2) - c


# f(xstar(c), c) = 0
def xstar(c):
    return c.pow(0.5)


def test_bisect():
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0

    options = {"lower": lower, "upper": upper, "dtype": torch.float64}

    root = beignet.root_scalar(
        f, c, method="bisect", implicit_diff=True, options=options
    )
    expected = xstar(c)

    torch.testing.assert_close(root, expected)


def test_bisect_grad():
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    options = {"lower": lower, "upper": upper, "dtype": torch.float64}

    grad = torch.func.vmap(
        torch.func.grad(
            lambda c: beignet.root_scalar(f, c, method="bisect", options=options)
        )
    )(c)

    expected = torch.func.vmap(torch.func.grad(xstar))(c)

    torch.testing.assert_close(grad, expected)

import beignet
import pytest
import torch


def f(x, c):
    return x.pow(2) - c


# f(xstar(c), c) = 0
def xstar(c):
    return c.pow(0.5)


@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar(method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0

    options = {"lower": lower, "upper": upper, "dtype": torch.float64}

    root = beignet.root_scalar(f, c, method=method, implicit_diff=True, options=options)
    expected = xstar(c)

    torch.testing.assert_close(root, expected)


@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar_grad(method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    options = {"lower": lower, "upper": upper, "dtype": torch.float64}

    grad = torch.func.vmap(
        torch.func.grad(
            lambda c: beignet.root_scalar(f, c, method=method, options=options)
        )
    )(c)

    expected = torch.func.vmap(torch.func.grad(xstar))(c)

    torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar_jacrev(method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    options = {"lower": lower, "upper": upper, "dtype": torch.float64}

    jac = torch.func.jacrev(
        lambda c: beignet.root_scalar(f, c, method=method, options=options)
    )(c)

    expected = torch.func.vmap(torch.func.grad(xstar))(c)

    torch.testing.assert_close(torch.diag(jac), expected)

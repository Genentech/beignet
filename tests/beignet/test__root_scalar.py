import platform
from functools import partial

import pytest
import torch

import beignet


def f(x, c):
    return x.pow(2) - c


# f(xstar(c), c) = 0
def xstar(c):
    return c.pow(0.5)


@pytest.mark.parametrize(
    "compile",
    [
        pytest.param(False, id="compile=True"),
        pytest.param(
            True,
            id="compile=False",
            marks=pytest.mark.skipif(
                platform.system() == "Windows",
                reason="torch.compile is broken on windows in github ci",
            ),
        ),
    ],
)
@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar(compile, method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    maxiter = 100

    options = {
        "a": lower,
        "b": upper,
        "dtype": torch.float64,
        "return_solution_info": True,
        "maxiter": maxiter,
    }

    solver = partial(
        beignet.root_scalar, method=method, implicit_diff=True, options=options
    )
    if compile:
        solver = torch.compile(solver, fullgraph=False)

    root, info = solver(f, c)

    expected = xstar(c)

    assert info["converged"].all()
    assert (info["iterations"] < maxiter).all()
    torch.testing.assert_close(root, expected)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="torch.compile is broken on windows in github ci",
)
@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar_compile_fullgraph(method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    maxiter = 100

    options = {
        "a": lower,
        "b": upper,
        "dtype": torch.float64,
        "return_solution_info": True,
        "maxiter": maxiter,
        "check_bracket": False,
    }

    solver = partial(
        beignet.root_scalar, method=method, implicit_diff=False, options=options
    )
    solver = torch.compile(solver, fullgraph=True)

    root, info = solver(f, c)

    expected = xstar(c)

    assert info["converged"].all()
    assert (info["iterations"] < maxiter).all()
    torch.testing.assert_close(root, expected)


@pytest.mark.parametrize("method", ["bisect", "chandrupatla"])
def test_root_scalar_grad(method):
    c = torch.linspace(1.0, 10.0, 101, dtype=torch.float64)

    lower = 0.0
    upper = 5.0
    options = {"a": lower, "b": upper, "dtype": torch.float64}

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
    options = {"a": lower, "b": upper, "dtype": torch.float64}

    jac = torch.func.jacrev(
        lambda c: beignet.root_scalar(f, c, method=method, options=options)
    )(c)

    expected = torch.func.vmap(torch.func.grad(xstar))(c)

    torch.testing.assert_close(torch.diag(jac), expected)

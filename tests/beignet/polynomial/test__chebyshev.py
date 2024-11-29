import torch
from beignet.polynomial import Chebyshev


def test_chebyshev_fit_1d():
    a = 0.0
    b = 2.0

    def f(x):
        return torch.exp(-x + torch.cos(10 * x))

    interp = Chebyshev.fit(f, d=1, a=[a], b=[b], order=200, dtype=torch.float64)

    x = torch.linspace(a, b, 1001, dtype=torch.float64)

    err = f(x) - interp(x)

    assert (err.abs() < 1e-10).all()


def test_chebyshev_fit_2d():
    def f(x, y):
        return torch.exp(-x * y + torch.cos(10 * x / (y.pow(2) + 1))) * torch.exp(
            -y.pow(2) + 1.0
        )

    interp = Chebyshev.fit(f, d=2, order=[201, 101], dtype=torch.float64)

    x = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, 501, dtype=torch.float64)

    err = f(x[:, None], y[None, :]) - interp(x, y)

    assert (err.abs() < 1e-10).all()


def test_chebyshev_cumulative_integral_2d():
    def f(x, y):
        return torch.sin(2 * x + 1) * y

    def int_f(x, y):
        return torch.sin(x) * torch.sin(x + 1) * y

    interp = Chebyshev.fit(f, d=2, order=[201, 101], dtype=torch.float64)

    int_interp = interp.cumulative_integral(dim=0)

    x = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, 501, dtype=torch.float64)

    err = int_f(x[:, None], y[None, :]) - int_interp(x, y)

    assert (err.abs() < 1e-10).all()

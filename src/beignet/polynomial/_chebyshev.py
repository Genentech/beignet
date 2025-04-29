import functools
from typing import Callable

import torch
from torch import Tensor


def _dct_ii_fft(input: Tensor, dim: int = -1):
    input = input.transpose(dim, -1)

    # see https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
    N = input.shape[-1]

    y = torch.zeros(*input.shape[:-1], 2 * N, device=input.device, dtype=input.dtype)
    y[..., :N] = input
    y[..., N:] = input.flip((-1,))

    output = torch.fft.fft(y)[..., :N]

    k = torch.arange(N, device=input.device, dtype=input.dtype)
    output *= torch.exp(-1j * torch.pi * k / (2 * N))

    output = output.transpose(dim, -1)

    return output.real


def _idct_ii_fft(input: Tensor, dim: int = -1):
    input = input.transpose(dim, -1)

    N = input.shape[-1]
    k = torch.arange(N, dtype=input.dtype, device=input.device)

    # Makhoul 9a
    yk_half = torch.exp(1j * torch.pi * k / (2 * N)) * input

    # Makhoul 12,13
    yk = torch.cat(
        [
            yk_half,
            torch.zeros(*input.shape[:-1], 1, dtype=input.dtype, device=input.device),
            yk_half[..., 1:].conj().flip((-1,)),
        ],
        dim=-1,
    )

    yn = torch.fft.ifft(yk)

    output = yn[..., :N].real

    return output.transpose(dim, -1)


@functools.lru_cache(maxsize=128)
def chebyshev_t_roots(N: int, device=None, dtype=None):
    k = torch.arange(N, device=device, dtype=dtype)
    xk = torch.cos(torch.pi * (2 * k + 1) / (2 * N))
    return xk


def chebyshev_t_values_to_coefficients(input: Tensor, dim: int = -1):
    input = input.transpose(dim, -1)

    N = input.shape[-1]
    j = torch.arange(0, N, dtype=input.dtype, device=input.device)
    delta_j0 = (j == 0).to(input.dtype)
    c = ((2 - delta_j0) / (2 * N)) * _dct_ii_fft(input)

    c = c.transpose(dim, -1)
    return c


def chebyshev_t_coefficients_to_values(input: Tensor):
    N = input.shape[-1]
    j = torch.arange(0, N, dtype=input.dtype, device=input.device)
    delta_j0 = (j == 0).to(input.dtype)
    return _idct_ii_fft(input * (2 * N / (2 - delta_j0)))


def chebyshev_t_clenshaw_eval(x: Tensor, a: Tensor):
    # x: shape (*,)
    # a: shape (**,N)
    # output: shape (*, **)

    N = a.shape[-1]

    x = x[..., None]

    bk_plus_one = torch.zeros_like(x)
    bk_plus_two = torch.zeros_like(x)

    for k in range(N - 1, 0, -1):
        bk = a[..., k] + 2 * x * bk_plus_one - bk_plus_two
        bk_plus_two = bk_plus_one
        bk_plus_one = bk

    # one more iteration to get b0
    bk = a[..., 0] + 2 * x * bk_plus_one - bk_plus_two

    return (0.5 * (a[..., 0] + bk - bk_plus_two)).squeeze(-1)


def from_chebyshev_domain(x, a, b):
    """Map x [-1, 1] to y in [a, b]."""
    return x * (b - a) / 2 + (b + a) / 2


def to_chebyshev_domain(y, a, b):
    """Map y in [a, b] to x in [-1,1]."""
    return (1 / (b - a)) * (2 * y - b - a)


def chebyshev_t_integral_operator(N: int, dtype=None, device=None):
    a = torch.cat(
        [
            torch.tensor(
                [
                    0.25,
                ],
                dtype=dtype,
                device=device,
            ),
            -0.5 * (1 / (torch.arange(2, N + 1, dtype=dtype, device=device) - 1)),
        ]
    )
    b = torch.cat(
        [
            torch.tensor([1, 0.25], dtype=dtype, device=device),
            0.5 * (1 / (torch.arange(2, N, dtype=dtype, device=device) + 1)),
        ]
    )
    op = (torch.diagflat(a, 1) + torch.diagflat(b, -1))[:, :-1]
    return op


def chebyshev_t_cumulative_integral_coefficients(
    coefficients: Tensor,
    a: float = -1.0,
    b: float = 1.0,
    dim: int = -1,
    complement: bool = False,
):
    """Compute the coefficients for the chebyshev series for the cummulative integral."""
    device = coefficients.device
    dtype = coefficients.dtype
    coefficients = coefficients.transpose(dim, -1)
    N = coefficients.shape[-1]
    op = chebyshev_t_integral_operator(N, device=device, dtype=dtype)
    c_int = torch.einsum("ij,...j->...i", op, coefficients)
    c_int *= (b - a) / 2

    if complement:
        int_b = c_int.sum(dim=-1)
        c_int = -1 * c_int
        c_int[..., 0] += int_b
    else:
        int_a = (c_int * (-1) ** torch.arange(N + 1, device=device, dtype=dtype)).sum(
            dim=-1
        )
        c_int[..., 0] -= int_a

    c_int = c_int.transpose(dim, -1)
    return c_int


class Chebyshev:
    """Multidimensional chebyshev series representation."""

    def __init__(
        self,
        coefficients: torch.Tensor,
        a: list[float] | None,
        b: list[float] | None,
    ):
        d = coefficients.ndim

        assert d >= 0

        if a is None:
            a = [-1] * d

        if b is None:
            b = [1] * d

        self.coefficients = coefficients
        self.d = d
        self.a = a
        self.b = b

    @classmethod
    def fit(
        cls,
        f: Callable,
        d: int,
        order: int | list[int],
        a: list[float] | None = None,
        b: list[float] | None = None,
        device=None,
        dtype=None,
    ):
        if isinstance(order, int):
            order = [order] * d

        if a is None:
            a = [-1] * d

        if b is None:
            b = [1] * d

        y = []
        for i in range(d):
            shape = [order[i] if j == i else 1 for j in range(d)]
            xi = chebyshev_t_roots(order[i], device=device, dtype=dtype)
            yi = from_chebyshev_domain(xi, a[i], b[i]).view(*shape)
            y.append(yi)

        values = f(*y)

        c = values
        for i in range(d):
            c = chebyshev_t_values_to_coefficients(c, dim=i)

        return cls(c, a, b)

    def __call__(self, *args):
        assert len(args) == self.d

        out = self.coefficients
        for i in range(self.d - 1, -1, -1):
            xi = to_chebyshev_domain(args[i], self.a[i], self.b[i])
            out = chebyshev_t_clenshaw_eval(xi, out)

        return out

    def cumulative_integral(self, dim: int = -1, complement: bool = False):
        c_int = chebyshev_t_cumulative_integral_coefficients(
            self.coefficients,
            a=self.a[dim],
            b=self.b[dim],
            dim=dim,
            complement=complement,
        )
        return Chebyshev(c_int, self.a, self.b)

import math

import torch
from torch import Tensor


def _voigt_v(x, y, N: int = 11):
    assert (x >= 0.0).all()
    assert (y >= 0.0).all()
    h = math.sqrt(math.pi / (N + 1))

    phi = (x / h) - (x / h).floor()

    k = torch.arange(N + 1, dtype=x.dtype, device=x.device)
    t = (k + 0.5) * h
    tau = k[1:] * h

    # equation 12
    w_m = (2 * h * y / torch.pi) * (
        torch.exp(-t.pow(2))
        * (t.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((t - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((t + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 13
    expy = torch.exp(-2 * torch.pi * y / h)
    w_mm_1 = (
        2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (torch.cos(2 * x * y) * expy + torch.cos(2 * torch.pi * x / h - 2 * x * y))
    ) / (expy + 1 / expy + 2 * torch.cos(2 * torch.pi * x / h))

    w_mm = w_mm_1 + w_m

    w_mt_1 = (
        2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (torch.cos(2 * x * y) * expy - torch.cos(2 * torch.pi * x / h - 2 * x * y))
    ) / (expy + 1 / expy - 2 * torch.cos(2 * torch.pi * x / h))

    w_mt_2 = (h * y) / (torch.pi * (x.pow(2) + y.pow(2)))

    w_mt_3 = (2 * h * y / torch.pi) * (
        torch.exp(-tau.pow(2))
        * (tau.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((tau - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((tau + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 14
    w_mt = w_mt_1 + w_mt_2 + w_mt_3

    return torch.where(
        y >= torch.maximum(x, torch.tensor(torch.pi / h)),
        w_m,
        torch.where((y < x) & (1 / 4 <= phi) & (phi <= 3 / 4), w_mt, w_mm),
    )


def _voigt_l(x, y, N: int = 11):
    assert (x >= 0.0).all()
    assert (y >= 0.0).all()
    h = math.sqrt(math.pi / (N + 1))

    phi = (x / h) - (x / h).floor()

    k = torch.arange(N + 1, dtype=x.dtype, device=x.device)
    t = (k + 0.5) * h
    tau = k[1:] * h

    w_m = (2 * h * x / torch.pi) * (
        torch.exp(-t.pow(2))
        * (-t.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((t - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((t + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 13
    expy = torch.exp(-2 * torch.pi * y / h)
    w_mm_1 = (
        -2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (torch.sin(2 * x * y) * expy - torch.sin(2 * torch.pi * x / h - 2 * x * y))
    ) / (expy + 1 / expy + 2 * torch.cos(2 * torch.pi * x / h))

    w_mm = w_mm_1 + w_m

    w_mt_1 = (
        -2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (torch.sin(2 * x * y) * expy + torch.sin(2 * torch.pi * x / h - 2 * x * y))
    ) / (expy + 1 / expy - 2 * torch.cos(2 * torch.pi * x / h))

    w_mt_2 = (h * x) / (torch.pi * (x.pow(2) + y.pow(2)))

    w_mt_3 = (2 * h * x / torch.pi) * (
        torch.exp(-tau.pow(2))
        * (-tau.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((tau - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((tau + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 14
    w_mt = w_mt_1 + w_mt_2 + w_mt_3

    return torch.where(
        y >= torch.maximum(x, torch.tensor(torch.pi / h)),
        w_m,
        torch.where((y < x) & (1 / 4 <= phi) & (phi <= 3 / 4), w_mt, w_mm),
    )


def faddeeva_w(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Faddeeva function.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
    """

    if not torch.is_complex(input):
        input = torch.complex(input, torch.zeros_like(input))

    # use symmetries to map to upper right quadrant of complex plane
    imag_negative = input.imag < 0.0
    input = torch.where(input.imag < 0.0, -input, input)
    real_negative = input.real < 0.0
    input = torch.where(input.real < 0.0, -input.conj(), input)

    x = input.real
    y = input.imag

    assert (x >= 0.0).all()
    assert (y >= 0.0).all()

    output = _voigt_v(x, y, N=11) + 1j * _voigt_l(x, y, N=11)

    # compute real and imaginary parts separately to so we handle infs
    # without unnecessary nans
    expz2 = torch.complex(
        2 * torch.exp(-x.pow(2) + y.pow(2)) * torch.cos(-2 * x * y),
        2 * torch.exp(-x.pow(2) + y.pow(2)) * torch.sin(-2 * x * y),
    )
    output = torch.where(imag_negative, expz2 - output, output)
    output = torch.where(real_negative, output.conj(), output, out=out)

    if out is not None:
        out.copy_(output)
        return out
    else:
        return output

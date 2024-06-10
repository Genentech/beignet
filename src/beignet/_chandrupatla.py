from typing import Callable, Optional

import torch
from torch import Tensor


def chandrupatla(
    f: Callable,
    x0: Tensor,
    x1: Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: int = 100,
    **_,
):
    b = x0
    a = x1
    c = x1
    fa = f(a)
    fb = f(b)
    fc = fa

    assert (torch.sign(fa) * torch.sign(fb) <= 0).all()

    t = 0.5 * torch.ones_like(fa)
    xm = torch.zeros_like(a)

    iterations = torch.zeros_like(fa, dtype=int)
    converged = torch.zeros_like(fa, dtype=bool)

    eps = torch.finfo(fa.dtype).eps
    if rtol is None:
        rtol = eps
    if atol is None:
        atol = 2 * eps

    for _ in range(maxiter):
        xt = a + t * (b - a)
        ft = f(xt)
        (
            a,
            b,
            c,
            fa,
            fb,
            fc,
            t,
            xt,
            ft,
            xm,
            converged,
            iterations,
        ) = _find_root_chandrupatla_iter(
            a, b, c, fa, fb, fc, t, xt, ft, xm, converged, iterations, atol, rtol
        )

        if converged.all():
            break

    return xm, (converged, iterations)


def _find_root_chandrupatla_iter(
    a, b, c, fa, fb, fc, t, xt, ft, xm, converged, iterations, atol, rtol
):
    cond = torch.sign(ft) == torch.sign(fa)
    c = torch.where(cond, a, b)
    fc = torch.where(cond, fa, fb)
    b = torch.where(cond, b, a)
    fb = torch.where(cond, fb, fa)

    a = xt
    fa = ft

    xm = torch.where(converged, xm, torch.where(torch.abs(fa) < torch.abs(fb), a, b))

    tol = 2 * rtol * torch.abs(xm) + atol
    tlim = tol / torch.abs(b - c)
    converged = converged | (tlim > 0.5)

    xi = (a - b) / (c - b)
    phi = (fa - fb) / (fc - fb)

    do_iqi = (phi.pow(2) < xi) & ((1 - phi).pow(2) < (1 - xi))

    t = torch.where(
        do_iqi,
        fa / (fb - fa) * fc / (fb - fc)
        + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb),
        0.5,
    )

    # limit to the range (tlim, 1-tlim)
    t = torch.minimum(1 - tlim, torch.maximum(tlim, t))

    iterations += ~converged

    return a, b, c, fa, fb, fc, t, xt, ft, xm, converged, iterations

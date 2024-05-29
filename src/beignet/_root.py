from typing import Callable, Optional

import torch


def root(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_iter: int = 100,
    method: str = "chandrupatla",
    **kwargs,
):
    """Find a root of a function.

    Parameters
    ----------
    f: Callable[[torch.Tensor], torch.Tensor]
        Function to find root of.

    x0: torch.Tensor
        Left bracket of root.

    x1: torch.Tensor
        Right bracket of root.

    rtol: float, optional
        Relative tolerance. Defaults to eps for input dtype.

    atol: float, optional
        Absolve tolerance. Defaults to 2*eps for input dtype.

    max_iter: int, optional
        Maximum number of solver iterations.

    method: str, optional
        Solver method to use. Defaults to 'chandrupatla'.
    """
    if method == "chandrupatla":
        return _find_root_chandrupatla(
            f, x0, x1, rtol=rtol, atol=atol, max_iter=max_iter, **kwargs
        )
    else:
        raise ValueError(f"unknown method {method}")


@torch.compile(fullgraph=True, dynamic=True)
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


# adapted from https://www.embeddedrelated.com/showarticle/855.php
def _find_root_chandrupatla(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_iter: int = 100,
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

    for _ in range(max_iter):
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

    return xm, {"converged": converged, "iterations": iterations}

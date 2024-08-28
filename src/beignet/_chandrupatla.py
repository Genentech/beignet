from typing import Callable

import torch
from torch import Tensor

from beignet._root_scalar import RootSolutionInfo


def chandrupatla(
    func: Callable,
    *args,
    lower: float,
    upper: float,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    return_solution_info: bool = False,
    dtype=None,
    device=None,
    **_,
) -> Tensor | tuple[Tensor, RootSolutionInfo]:
    # maintain three points a,b,c for inverse quadratic interpolation
    # we will keep (a,b) as the bracketing interval
    a = torch.as_tensor(lower, dtype=dtype, device=device)
    b = torch.as_tensor(upper, dtype=dtype, device=device)
    a, b, *args = torch.broadcast_tensors(a, b, *args)
    c = a

    fa = func(a, *args)
    fb = func(b, *args)
    fc = fa

    # root estimate
    xm = torch.where(torch.abs(fa) < torch.abs(fb), a, b)

    eps = torch.finfo(fa.dtype).eps

    if rtol is None:
        rtol = eps

    if atol is None:
        atol = 2 * eps

    converged = torch.zeros_like(fa, dtype=torch.bool)
    iterations = torch.zeros_like(fa, dtype=torch.int)

    if (torch.sign(fa) * torch.sign(fb) > 0).any():
        raise ValueError("a and b must bracket a root")

    for _ in range(maxiter):
        xm = torch.where(
            converged, xm, torch.where(torch.abs(fa) < torch.abs(fb), a, b)
        )
        tol = atol + torch.abs(xm) * rtol
        bracket_size = torch.abs(b - a)
        tlim = tol / bracket_size
        #        converged = converged | 0.5 * bracket_size < tol
        converged = converged | (tlim > 0.5)

        if converged.all():
            break

        a, b, c, fa, fb, fc = _find_root_chandrupatla_iter(
            func, *args, a=a, b=b, c=c, fa=fa, fb=fb, fc=fc, tlim=tlim
        )

        iterations = iterations + ~converged

    if return_solution_info:
        return xm, RootSolutionInfo(converged=converged, iterations=iterations)
    else:
        return xm


def _find_root_chandrupatla_iter(
    func: Callable,
    *args,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    fa: Tensor,
    fb: Tensor,
    fc: Tensor,
    tlim: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # check validity of inverse quadratic interpolation
    xi = (a - b) / (c - b)
    phi = (fa - fb) / (fc - fb)
    do_iqi = (phi.pow(2) < xi) & ((1 - phi).pow(2) < (1 - xi))

    # use iqi where applicable, otherwise bisect interval
    t = torch.where(
        do_iqi,
        fa / (fb - fa) * fc / (fb - fc)
        + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb),
        0.5,
    )
    t = torch.clip(t, min=tlim, max=1 - tlim)

    xt = a + t * (b - a)
    ft = func(xt, *args)

    # check which side of root t is on
    cond = torch.sign(ft) == torch.sign(fa)

    # update a,b,c maintaining (a,b) a bracket of root
    # NOTE we do not maintain the order of a and b
    c = torch.where(cond, a, b)
    fc = torch.where(cond, fa, fb)
    b = torch.where(cond, b, a)
    fb = torch.where(cond, fb, fa)
    a = xt
    fa = ft

    return a, b, c, fa, fb, fc

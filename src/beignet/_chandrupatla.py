from typing import Callable

import torch
from torch import Tensor
from torch._higher_order_ops import while_loop


def chandrupatla(
    func: Callable,
    *args,
    a: float | Tensor,
    b: float | Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    return_solution_info: bool = False,
    check_bracket: bool = True,
    unroll: int = 1,
    **_,
) -> Tensor | tuple[Tensor, dict]:
    """Find the root of a scalar (elementwise) function using chandrupatla method.

    This method uses inverse quadratic interpolation to accelerate convergence.
    Like bisection it is guaranteed to converge.

    Parameters
    ----------
    func: Callable
        Function to find a root of. Called as `f(x, *args)`.
        The function must operate element wise, i.e. `f(x[i]) == f(x)[i]`.
        Handling *args via broadcasting is acceptable.

    *args
        Extra arguments to be passed to `func`.

    a: float | Tensor
        Lower bracket for root

    b: float | Tensor
        Upper bracket for root

    rtol: float | None = None
        Relative tolerance

    atol: float | None = None
        Absolute tolerance

    maxiter: int = 100
        Maximum number of iterations

    return_solution_info: bool = False
        Return a solution metadata dictionary.

    check_bracket: bool = True
        Check if input bracket is valid

    unroll: int = 1
        Number of iterations between convergence checks.
        Inside `torch.compile` these are unrolled which reduces kernel launch overhead.

    Returns
    -------
    Tensor | tuple[Tensor, dict]


    References
    ----------

    [1] Tirupathi R. Chandrupatla. A new hybrid quadratic/bisection algorithm for
    finding the zero of a nonlinear function without using derivatives.
    Advances in Engineering Software, 28.3:145-149, 1997.
    """
    # maintain three points a,b,c for inverse quadratic interpolation
    # we will keep (a,b) as the bracketing interval
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    a, b, *args = torch.broadcast_tensors(a, b, *args)

    dtype = a.dtype
    for x in (b, *args):
        dtype = torch.promote_types(x.dtype, dtype)

    eps = torch.finfo(dtype).eps
    a, b, *args = (x.to(dtype=dtype).contiguous() for x in (a, b, *args))

    c = a.clone()

    fa = func(a, *args)
    fb = func(b, *args)
    fc = fa.clone()

    # root estimate
    xm = torch.where(torch.abs(fa) < torch.abs(fb), a, b)

    if rtol is None:
        rtol = eps

    if atol is None:
        atol = 2 * eps

    converged = torch.zeros_like(fa, dtype=torch.bool)
    iterations = torch.zeros_like(fa, dtype=torch.int)

    if check_bracket and (torch.sign(fa) * torch.sign(fb) > 0).any():
        raise ValueError("a and b must bracket a root")

    def condition(a, b, c, fa, fb, fc, xm, converged, iterations):
        return ~converged.all() & (iterations <= maxiter).all()

    def loop_body(a, b, c, fa, fb, fc, xm, converged, iterations):
        for _ in range(unroll):
            tol = atol + torch.abs(xm) * rtol
            bracket_size = torch.abs(b - a)
            tlim = tol / bracket_size
            converged = converged | (tlim > 0.5)

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
            a = xt.clone()
            fa = ft.clone()

            xm = torch.where(
                converged, xm, torch.where(torch.abs(fa) < torch.abs(fb), a, b)
            )

            iterations = iterations + ~converged
        return (
            a,
            b,
            c,
            fa,
            fb,
            fc,
            xm,
            converged,
            iterations,
        )

    a, b, c, fa, fb, fc, xm, converged, iterations = while_loop(
        condition, loop_body, (a, b, c, fa, fb, fc, xm, converged, iterations)
    )

    if return_solution_info:
        return xm, {"converged": converged, "iterations": iterations}
    else:
        return xm

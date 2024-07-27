from functools import wraps
from typing import Callable

import torch
from torch import Tensor


def multiplicative_isotropic_cutoff(
    fn: Callable[..., Tensor], r_onset: Tensor, r_cutoff: Tensor
) -> Callable[..., Tensor]:
    r"""Takes an isotropic function and constructs a truncated function.

    Given a function `f:R -> R`, we construct a new function `f':R -> R` such
    that `f'(r) = f(r)` for `r < r_onset`, `f'(r) = 0` for `r > r_cutoff`, and
    `f(r)` is :math:`C^1` everywhere. To do this, we follow the approach outlined
    in HOOMD Blue  [#hoomd]_ (thanks to Carl Goodrich for the pointer). We
    construct a function `S(r)` such that `S(r) = 1` for `r < r_onset`,
    `S(r) = 0` for `r > r_cutoff`, and `S(r)` is :math:`C^1`. Then
    `f'(r) = S(r)f(r)`.

    Parameters
    ----------
    fn : Callable
        A function that takes a tensor of distances of shape `[n, m]` as well
        as varargs.
    r_onset : Tensor
        A Tensor specifying the distance marking the onset of deformation.
    r_cutoff : Tensor
        A Tensor specifying the cutoff distance.

    Returns
    -------
    callable
        A new function with the same signature as fn, with the properties outlined
        above.
    """

    r_c = r_cutoff**2
    r_o = r_onset**2

    def smooth_fn(dr):
        r = dr**2

        inner = torch.where(
            dr < r_cutoff,
            (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o) ** 3,
            torch.zeros_like(dr),
        )

        return torch.where(dr < r_onset, torch.ones_like(dr), inner)

    @wraps(fn)
    def cutoff_fn(dr, *args, **kwargs):
        smooth = smooth_fn(dr)
        func = fn(dr, *args, **kwargs)
        return smooth_fn(dr) * fn(dr, *args, **kwargs)

    return cutoff_fn

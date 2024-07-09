import math

import torch
from torch import Tensor

from ._physicists_hermite_polynomial_vandermonde import (
    physicists_hermite_polynomial_vandermonde,
)


def fit_physicists_hermite_polynomial(
    input: Tensor,
    other: Tensor,
    degree: Tensor | int,
    relative_condition: float | None = None,
    full: bool = False,
    weight: Tensor | None = None,
):
    input = torch.tensor(input)
    other = torch.tensor(other)
    degree = torch.tensor(degree)
    if degree.ndim > 1:
        raise TypeError
    # if deg.dtype.kind not in "iu":
    #     raise TypeError
    if math.prod(degree.shape) == 0:
        raise TypeError
    if degree.min() < 0:
        raise ValueError
    if input.ndim != 1:
        raise TypeError
    if input.size == 0:
        raise TypeError
    if other.ndim < 1 or other.ndim > 2:
        raise TypeError
    if len(input) != len(other):
        raise TypeError
    if degree.ndim == 0:
        lmax = int(degree)
        van = physicists_hermite_polynomial_vandermonde(input, lmax)
    else:
        degree, _ = torch.sort(degree)
        lmax = int(degree[-1])
        van = physicists_hermite_polynomial_vandermonde(input, lmax)[:, degree]
    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = other.T
    if weight is not None:
        if weight.ndim != 1:
            raise TypeError("expected 1D vector for w")

        if len(input) != len(weight):
            raise TypeError("expected x and w to have same length")

        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * weight
        rhs = rhs * weight
    # set rcond
    if relative_condition is None:
        relative_condition = len(input) * torch.finfo(input.dtype).eps
    # Determine the norms of the design matrix columns.
    if torch.is_complex(lhs):
        scl = torch.sqrt((torch.square(lhs.real) + torch.square(lhs.imag)).sum(1))
    else:
        scl = torch.sqrt(torch.square(lhs).sum(1))
    scl = torch.where(scl == 0, 1, scl)
    # Solve the least squares problem.
    c, resids, rank, s = torch.linalg.lstsq(lhs.T / scl, rhs.T, relative_condition)
    c = (c.T / scl).T
    # Expand c to include non-fitted coefficients which are set to zero
    if degree.ndim > 0:
        if c.ndim == 2:
            cc = torch.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = torch.zeros(lmax + 1, dtype=c.dtype)

        cc[degree] = c

        c = cc
    if full:
        result = c, [resids, rank, s, relative_condition]
    else:
        result = c
    return result

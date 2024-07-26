import torch
from torch import Tensor


def legendre_polynomial_vandermonde(x: Tensor, degree: Tensor) -> Tensor:
    if degree < 0:
        raise ValueError

    x = torch.tensor(x)
    x = torch.atleast_1d(x)

    dims = (degree + 1,) + x.shape

    dtype = torch.promote_types(x.dtype, torch.tensor(0.0).dtype)

    x = x.to(dtype)

    v = torch.empty(dims, dtype=dtype)

    v[0] = torch.ones_like(x)

    if degree > 0:
        v[1] = x

        for index in range(2, degree + 1):
            v[index] = (
                v[index - 1] * x * (2 * index - 1) - v[index - 2] * (index - 1)
            ) / index

    return torch.moveaxis(v, 0, -1)

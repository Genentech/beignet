import torch
from torch import Tensor


def laguerre_polynomial_vandermonde(
    x: Tensor,
    degree: Tensor,
) -> Tensor:
    if degree < 0:
        raise ValueError

    x = torch.atleast_1d(x)

    dtype = torch.promote_types(x.dtype, torch.get_default_dtype())

    x = x.to(dtype)

    v = torch.empty([degree + 1, *x.shape], dtype=dtype)

    v[0] = torch.ones_like(x)

    if degree > 0:
        v[1] = 1 - x

        for index in range(2, degree + 1):
            v[index] = (
                v[index - 1] * (2 * index - 1 - x) - v[index - 2] * (index - 1)
            ) / index

    return torch.moveaxis(v, 0, -1)

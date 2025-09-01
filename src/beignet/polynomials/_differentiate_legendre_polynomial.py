import torch
from torch import Tensor


def differentiate_legendre_polynomial(
    input,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    r"""
    Returns the derivative of a polynomial.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    order : Tensor, optional

    scale : Tensor, optional

    dim : int, default=0

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the derivative.
    """
    if order < 0:
        raise ValueError

    input = torch.atleast_1d(input)

    if order == 0:
        return input

    input = torch.moveaxis(input, axis, 0)

    n = input.shape[0]

    if order >= n:
        input = torch.zeros_like(input[:1])
    else:
        for _ in range(order):
            n = n - 1
            input *= scale
            der = torch.empty((n,) + input.shape[1:], dtype=input.dtype)

            def body(k, der_c, n=n):
                j = n - k

                der, c = der_c

                der[j - 1] = (2 * j - 1) * c[j]

                c[j - 2] += c[j]

                return der, c

            b = n - 2

            x = (der, input)

            y = x

            for index in range(0, b):
                y = body(index, y)

            der, input = y

            if n > 1:
                der[1] = 3 * input[2]

            der[0] = input[1]

            input = der

    input = torch.moveaxis(input, 0, axis)

    return input

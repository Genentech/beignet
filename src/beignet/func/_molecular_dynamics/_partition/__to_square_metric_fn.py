from typing import Any, Callable

import torch
from torch import Tensor


def _to_square_metric_fn(
    fn: Callable[[Tensor, Tensor, Any], Tensor],
) -> Callable[[Tensor, Tensor, Any], Tensor]:
    r"""Converts a given distance function to a squared distance metric.

    The function tries to apply the given distance function `fn` to positions in
    one to three dimensions to determine if the output is scalar or vector.
    Based on this, it returns a new function that computes the squared distance.

    Parameters
    ----------
    fn : Callable[[Tensor, Tensor, Any], Tensor]
        A function that computes distances between two tensors.

    Returns
    -------
    Callable[[Tensor, Tensor, Any], Tensor]
        A function that computes the squared distance metric using the given distance function.
    """
    for dimension in range(1, 4):
        try:
            positions = torch.rand([dimension], dtype=torch.float32)

            distances = fn(positions, positions, t=0)  # type: ignore[no-untyped-def]

            if distances.ndim == 0:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.square(fn(a, b, **kwargs))
            else:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.sum(torch.square(fn(a, b, **kwargs)), dim=-1)

            return square_metric

        except TypeError:
            continue

        except ValueError:
            continue

    raise ValueError

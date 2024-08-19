from typing import Callable

import torch


def map_neighbor(metric_or_displacement: Callable) -> Callable:
    r"""Vectorizes a metric or displacement function over neighborhoods.

    Parameters
    ----------
    metric_or_displacement : callable
        A function that computes a metric or displacement between two inputs.
        This function should accept two arguments and return a single value
         representing the metric or displacement.

    Returns
    -------
    wrapped_fn : callable
        A vectorized function that applies `metric_or_displacement` over
        neighborhoods of input data. The returned function takes two arguments:
        `input` and `other`, where `input` is the reference data and `other` is
        the neighborhood data.
    """
    def wrapped_fn(input, other, **kwargs):
        return torch.vmap(torch.vmap(metric_or_displacement, (0, None)))(
            other, input, **kwargs
        )

    return wrapped_fn
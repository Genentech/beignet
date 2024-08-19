from typing import Callable

import torch


def map_product(metric_or_displacement: Callable) -> Callable:
    r"""Vectorizes a metric or displacement function over all pairs.

    Parameters
    ----------
    metric_or_displacement : callable
        A function that computes a metric or displacement between two inputs.
        This function should accept two arguments and return a single value
        representing the metric or displacement.

    Returns
    -------
    wrapped_fn : callable
        A vectorized function that applies `metric_or_displacement` over all
        pairs of input data. The returned function takes two arguments:
        `input1` and `input2`, where `input1` and `input2` are the sets of data
         to be compared.
    """
    return torch.vmap(torch.vmap(metric_or_displacement, (0, None)), (None, 0))
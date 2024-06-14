from typing import Callable

import torch


def metric(distance_fn: Callable) -> Callable:
    r"""Takes a displacement function and creates a metric..

    Parameters:
    -----------
    distance_fn : callable
        A function that computes the distance between two positions.

    Returns:
    --------
    wrapper : callable
        A wrapper function that applies `distance_fn` to each pair of start and end positions
        in the batch.

    Example:
    --------
    >>> # Assume `distance_fn` computes the Euclidean distance between two points
    >>> start_positions = torch.tensor([[0, 0], [1, 1]])
    >>> end_positions = torch.tensor([[0, 1], [1, 2]])
    >>> wrapped_fn = _map_bond(distance_fn)
    >>> result = wrapped_fn(start_positions, end_positions)
    >>> print(result)
    tensor([...])
    """
    def wrapper(start_positions, end_positions):
        batch_size = start_positions.shape[0]

        return torch.stack(
            [
                distance_fn(start_positions[i], end_positions[i])
                for i in range(batch_size)
            ]
        )

    return wrapper
import heapq
from typing import Callable

import torch
from torch import Tensor


def farthest_first_traversal(
    input: Tensor,
    distance_func: Callable,
    scores: Tensor | None = None,
    n: int = 2,
    descending: bool = False,
) -> Tensor:
    """
    Farthest-first traversal of a library of N elements.
    If `ranking_scores` is provided, the scores are used to pick the starting point
    and to break ties based on edit distance.
    If no scores are provided, ties are broken by the library index.
    Args:
        input: A list with N elements or a tensor with shape (N, *element_shape).
        distance_func: A callable that takes two arguments and
            returns a distance between them.
        scores: A tensor with shape (N,) containing the ranking scores
            of the elements in the library.
        n: The number of library elements to return.
        descending: If False and ranking_scores is provided, the elements with
            the lowest scores are selected first in a tie (default: False).
    Returns:
        A tensor with shape (n,) containing the indices of the selected elements.
    """
    if scores is None:
        remaining_indices = list(range(len(input)))
    else:
        if descending:
            scores = -scores

        remaining_indices = list(torch.argsort(scores))

    selected = [remaining_indices.pop(0)]

    if n == 1:
        return torch.tensor(selected)

    queue = []

    # First pass through library
    for index in remaining_indices:
        # Pushing with heapq, negate dist to simulate max-heap with min-heap
        neg_dist = -distance_func(input[index], input[selected[0]])

        if scores is None:
            item = (neg_dist, index, 1)
        else:
            item = (neg_dist, scores[index], index, 1)

        heapq.heappush(queue, item)

    for _ in range(1, n):
        item = heapq.heappop(queue)

        while True:
            if scores is None:
                neg_dist, idx, num_checked = item
            else:
                neg_dist, score, idx, num_checked = item

            # Check if the top of the heap has been checked
            # against all currently selected sequences
            if num_checked < len(selected):
                min_dist = min(
                    distance_func(input[idx], input[selected[i]])
                    for i in range(num_checked, len(selected))
                )

                min_dist = min(min_dist, -neg_dist)

                if scores is None:
                    item = (-min_dist, idx, len(selected))
                else:
                    item = (-min_dist, score, idx, len(selected))

                item = heapq.heappushpop(queue, item)
            else:
                selected.append(idx)

                break

    return torch.tensor(selected)

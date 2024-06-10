import heapq
from typing import Any, Iterable

import torch


def farthest_first_traversal(
    library: Iterable[Any],
    distance_fn: callable,
    ranking_scores: torch.Tensor | None = None,
    n: int = 2,
    descending: bool = False,
) -> torch.Tensor:
    """
    Farthest-first traversal of a library of strings.
    If `ranking_scores` is provided, the scores are used to pick the starting point
    and to break ties based on edit distance.
    If no scores are provided, ties are broken by the library index.

    Args:
        library: A numpy array of shape (N,) where N is the number of sequences.
        distance_fn: A callable that takes two arguments and
            returns a distance between them.
        ranking_scores: A tensor with shape (N,) containing the ranking scores
            of the sequences in the library.
        n: The number of sequences to return.

    Returns:
        A tensor with shape (n,) containing the indices of the selected sequences.
    """
    if ranking_scores is None:
        ranking_scores = torch.zeros(len(library))
        remaining_indices = list(range(len(library)))
    else:
        if descending:
            ranking_scores = -ranking_scores
        remaining_indices = list(torch.argsort(ranking_scores))

    selected = [remaining_indices.pop(0)]

    if n == 1:
        return torch.tensor(selected)

    pq = []
    # First pass through library
    for index in remaining_indices:
        # Pushing with heapq, negate dist to simulate max-heap with min-heap
        (
            heapq.heappush(
                pq,
                (
                    -distance_fn(library[index], library[selected[0]]),
                    ranking_scores[index],
                    index,
                    1,
                ),
            ),
        )

    for _ in range(1, n):
        while True:
            neg_dist, score, idx, num_checked = heapq.heappop(pq)
            # Check if the top of the heap has been checked
            # against all currently selected sequences
            if num_checked < len(selected):
                min_dist = min(
                    distance_fn(library[idx], library[selected[i]])
                    for i in range(num_checked, len(selected))
                )
                min_dist = min(min_dist, -neg_dist)
                heapq.heappush(pq, (-min_dist, score, idx, len(selected)))
            else:
                selected.append(idx)
                break

    return torch.tensor(selected)

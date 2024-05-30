import edlib
import heapq
import numpy as np
from typing import Optional


def edit_dist(x: str, y: str):
    """
    Computes the edit distance between two strings.
    """
    return edlib.align(x, y)["editDistance"]


def ranked_fft(
    library: np.ndarray,
    ranking_scores: Optional[np.ndarray] = None,
    n: int = 2,
    descending: bool = False,
):
    """
    Farthest-first traversal of a library of strings.
    If `ranking_scores` is provided, the scores are used to pick the starting point and break ties.

    Args:
        library: A numpy array of shape (N,) where N is the number of sequences.
        ranking_scores: A numpy array of shape (N,) containing the ranking scores of the sequences in the library.
        n: The number of sequences to return.

    Returns:
        A numpy array of shape (n,) containing the indices of the selected sequences.
    """
    if ranking_scores is None:
        ranking_scores = np.zeros(library.shape[0])
        remaining_indices = list(range(library.shape[0]))
    else:
        if descending:
            ranking_scores = -ranking_scores
        remaining_indices = list(np.argsort(ranking_scores))

    selected = [remaining_indices.pop(0)]

    if n == 1:
        return np.array(selected)

    pq = []
    # First pass through library
    for index in remaining_indices:
        # Pushing with heapq, negate dist to simulate max-heap with min-heap
        (
            heapq.heappush(
                pq,
                (
                    -edit_dist(library[index], library[selected[0]]),
                    ranking_scores[index],
                    index,
                    1,
                ),
            ),
        )

    for _ in range(1, n):
        while True:
            neg_dist, score, idx, num_checked = heapq.heappop(pq)
            # Check if the top of the heap has been checked against all currently selected sequences
            if num_checked < len(selected):
                min_dist = min(
                    edit_dist(library[idx], library[selected[i]])
                    for i in range(num_checked, len(selected))
                )
                min_dist = min(min_dist, -neg_dist)
                heapq.heappush(pq, (-min_dist, score, idx, len(selected)))
            else:
                selected.append(idx)
                break

    return np.array(selected)

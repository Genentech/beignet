import torch
from beignet import farthest_first_traversal


def str_hamming_dist(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2, strict=False))


def test_farthest_first_traversal():
    library = [
        "AAAA",
        "GGGG",
        "CCCC",
        "TTTT",
    ]
    ranking_scores = torch.tensor([3, 2, 1, 4])
    n = 2
    selected = farthest_first_traversal(
        library, str_hamming_dist, ranking_scores, n, descending=True
    )

    assert torch.all(selected == torch.tensor([3, 0]))

    selected = farthest_first_traversal(
        library, str_hamming_dist, ranking_scores, n, descending=False
    )
    assert torch.all(selected == torch.tensor([2, 1]))

    ranking_scores = None
    selected = farthest_first_traversal(library, str_hamming_dist, ranking_scores, n)
    assert torch.all(selected == torch.tensor([0, 1]))

    # harder example with wider spread of distances
    library = [
        "AAAA",
        "GGGG",
        "CCCC",
        "TTTT",
        "ACGT",
        "TGCA",
        "ACGT",
        "TGCA",
    ]

    ranking_scores = torch.tensor(list(range(8)))
    n = 3
    selected = farthest_first_traversal(
        library, str_hamming_dist, ranking_scores, n, descending=True
    )
    assert torch.all(selected == torch.tensor([7, 6, 3]))

    selected = farthest_first_traversal(
        library, str_hamming_dist, ranking_scores, n, descending=False
    )
    assert torch.all(selected == torch.tensor([0, 1, 2]))

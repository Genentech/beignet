import beignet
import torch
import torch.nested
from torch import Tensor


def test_farthest_first_traversal():
    def distance_func(input, other):
        return sum(c1 != c2 for c1, c2 in zip(input, other, strict=False))

    def map_sequence(sequence: str) -> Tensor:
        dictionary = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
        }

        output = []

        for character in sequence:
            output = [*output, dictionary[character]]

        return torch.tensor([*output])

    sequences = [
        "AAAA",
        "GGGG",
        "CCCC",
        "TTTT",
    ]

    input = [map_sequence(sequence) for sequence in sequences]

    input = torch.nested.nested_tensor(input)

    input = torch.nested.to_padded_tensor(input, 0)

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            input,
            distance_func=distance_func,
            scores=torch.tensor([3, 2, 1, 4]),
            n=2,
            descending=True,
        ),
        torch.tensor([3, 0]),
    )

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            input,
            distance_func=distance_func,
            scores=torch.tensor([3, 2, 1, 4]),
            n=2,
            descending=False,
        ),
        torch.tensor([2, 1]),
    )

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            input,
            distance_func=distance_func,
            n=2,
        ),
        torch.tensor([0, 1]),
    )

    sequences = [
        "AAAA",
        "GGGG",
        "CCCC",
        "TTTT",
        "ACGT",
        "TGCA",
        "ACGT",
        "TGCA",
    ]

    input = [map_sequence(sequence) for sequence in sequences]

    input = torch.nested.nested_tensor(input)

    input = torch.nested.to_padded_tensor(input, 0)

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            input,
            distance_func=distance_func,
            scores=torch.tensor([*range(8)]),
            n=3,
            descending=True,
        ),
        torch.tensor([7, 6, 3]),
    )

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            input,
            distance_func=distance_func,
            scores=(torch.tensor(list(range(8)))),
            n=3,
            descending=False,
        ),
        torch.tensor([0, 1, 2]),
    )

    def distance_fn(input: Tensor, other: Tensor) -> Tensor:
        return torch.norm(input - other, p=2)

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 4.0],
                    [4.0, 0.0],
                    [2.0, 2.0],
                ]
            ),
            distance_func=distance_fn,
            scores=torch.tensor([3, 2, 1, 4]),
            n=2,
            descending=True,
        ),
        torch.tensor([3, 0]),
    )

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 4.0],
                    [4.0, 0.0],
                    [2.0, 2.0],
                ]
            ),
            distance_func=distance_fn,
            scores=(torch.tensor([3, 2, 1, 4])),
            n=2,
            descending=False,
        ),
        torch.tensor([2, 1]),
    )

    torch.testing.assert_close(
        beignet.farthest_first_traversal(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 4.0],
                    [4.0, 0.0],
                    [2.0, 2.0],
                ]
            ),
            distance_func=distance_fn,
            scores=None,
            n=2,
        ),
        torch.tensor([0, 1]),
    )

import random

import torch

import beignet


class FarthestFirstTraversalBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.farthest_first_traversal,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.library = torch.randn(batch_size, dtype=dtype)

        self.distance_fn = torch.rand(batch_size, dtype=dtype) * 10.0 + 0.1

        self.ranking_scores = torch.randn(batch_size, dtype=dtype)

        self.n = random.randint(1, 10)

        self.descending = random.choice([True, False])

    def time_farthest_first_traversal(self, batch_size, dtype):
        self.func(
            self.input,
            self.library,
            self.distance_fn,
            self.ranking_scores,
            self.n,
            self.descending,
        )

    def peak_memory_farthest_first_traversal(self, batch_size, dtype):
        self.func(
            self.input,
            self.library,
            self.distance_fn,
            self.ranking_scores,
            self.n,
            self.descending,
        )

import torch

import beignet

from ._set_seed import set_seed


class BenchRadiusGraph:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.radius_graph,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.r = torch.randn(batch_size, dtype=dtype)

        self.batch = torch.zeros(batch_size, dtype=torch.long)

        self.chunk_size = 1024

        self.loop = True

    def time_radius_graph(self, batch_size, dtype):
        self.func(self.x, self.r, self.batch, self.chunk_size, self.loop)

    def peak_memory_radius_graph(self, batch_size, dtype):
        self.func(self.x, self.r, self.batch, self.chunk_size, self.loop)

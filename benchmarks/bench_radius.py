import torch

import beignet


class BenchRadius:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.radius,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.r = torch.randn(batch_size, dtype=dtype)

        self.batch_x = torch.zeros(batch_size, dtype=torch.long)

        self.batch_y = torch.zeros(batch_size, dtype=torch.long)

        self.ignore_same_index = True

        self.chunk_size = 1024

    def time_radius(self, batch_size, dtype):
        self.func(
            self.x,
            self.y,
            self.r,
            self.batch_x,
            self.batch_y,
            self.ignore_same_index,
            self.chunk_size,
        )

    def peak_memory_radius(self, batch_size, dtype):
        self.func(
            self.x,
            self.y,
            self.r,
            self.batch_x,
            self.batch_y,
            self.ignore_same_index,
            self.chunk_size,
        )

import random

import torch

import beignet


class BenchKabsch:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.kabsch,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.weights = torch.randn(batch_size, dtype=dtype)

        self.driver = torch.randn(batch_size, dtype=dtype)

        self.keepdim = random.choice([True, False])

    def time_kabsch(self, batch_size, dtype):
        self.func(self.input, self.x, self.y, self.weights, self.driver, self.keepdim)

    def peak_memory_kabsch(self, batch_size, dtype):
        self.func(self.input, self.x, self.y, self.weights, self.driver, self.keepdim)

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchInvertTransform:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.invert_transform,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.transform = torch.randn(batch_size, dtype=dtype)

    def time_invert_transform(self, batch_size, dtype):
        self.func(self.transform)

    def peak_memory_invert_transform(self, batch_size, dtype):
        self.func(self.transform)

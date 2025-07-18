import torch

import beignet

from ._set_seed import set_seed


class BenchApplyTransform:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.apply_transform,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.transform = torch.randn(batch_size, dtype=dtype)

    def time_apply_transform(self, batch_size, dtype):
        self.func(self.input, self.transform)

    def peak_memory_apply_transform(self, batch_size, dtype):
        self.func(self.input, self.transform)

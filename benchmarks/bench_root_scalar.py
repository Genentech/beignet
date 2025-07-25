import torch

import beignet

from ._set_seed import set_seed


class BenchRootScalar:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.root_scalar,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, dtype=dtype)

        self.method = torch.randn(batch_size, dtype=dtype)

        self.implicit_diff = True

        self.options = torch.randn(batch_size, dtype=dtype)

    def time_root_scalar(self, batch_size, dtype):
        self.func(self.input, self.func, self.method, self.implicit_diff, self.options)

    def peak_memory_root_scalar(self, batch_size, dtype):
        self.func(self.input, self.func, self.method, self.implicit_diff, self.options)

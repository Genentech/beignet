import torch

import beignet

from ._set_seed import set_seed


class BenchOptionalDependencies:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.optional_dependencies,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.names = torch.randn(batch_size, dtype=dtype)

        self.groups = torch.randn(batch_size, dtype=dtype)

    def time_optional_dependencies(self, batch_size, dtype):
        self.func(self.names, self.groups)

    def peak_memory_optional_dependencies(self, batch_size, dtype):
        self.func(self.names, self.groups)

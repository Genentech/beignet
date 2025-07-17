import torch

import beignet

from ._set_seed import set_seed


class BenchDefaultDtypeManager:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.default_dtype_manager,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.dtype = dtype

    def time_default_dtype_manager(self, batch_size, dtype):
        with self.func(self.dtype):
            pass

    def peak_memory_default_dtype_manager(self, batch_size, dtype):
        with self.func(self.dtype):
            pass

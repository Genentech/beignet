import torch

import beignet


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
        self.dtype = torch.randn(batch_size, dtype=dtype)

    def time_default_dtype_manager(self, batch_size, dtype):
        self.func(self.input, self.dtype)

    def peak_memory_default_dtype_manager(self, batch_size, dtype):
        self.func(self.input, self.dtype)

import random

import torch

import beignet

from .._set_seed import set_seed


class BenchErrorERFC:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.error_erfc,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

    def time_error_erfc(self, batch_size, dtype):
        self.func(self.input, self.out)

    def peak_memory_error_erfc(self, batch_size, dtype):
        self.func(self.input, self.out)

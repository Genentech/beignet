import random

import torch

import beignet

from .._set_seed import set_seed


class BenchErrorERFI:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.special.error_erfi,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, dtype=dtype)

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

    def time_error_erfi(self, batch_size, dtype):
        self.func(self.input, self.out)

    def peak_memory_error_erfi(self, batch_size, dtype):
        self.func(self.input, self.out)

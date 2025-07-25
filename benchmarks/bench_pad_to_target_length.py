import random

import torch

import beignet

from ._set_seed import set_seed


class BenchPadToTargetLength:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.pad_to_target_length,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.target_length = 10

        self.dim = random.randint(-1, 2)

        self.value = torch.randn(batch_size, dtype=dtype)

    def time_pad_to_target_length(self, batch_size, dtype):
        self.func(self.input, self.target_length, self.dim, self.value)

    def peak_memory_pad_to_target_length(self, batch_size, dtype):
        self.func(self.input, self.target_length, self.dim, self.value)

import random

import torch

import beignet

from ._set_seed import set_seed


class BenchQuaternionMean:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.quaternion_mean,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.weight = random.choice([None, torch.randn(batch_size, dtype=dtype)])

    def time_quaternion_mean(self, batch_size, dtype):
        self.func(self.input, self.weight)

    def peak_memory_quaternion_mean(self, batch_size, dtype):
        self.func(self.input, self.weight)

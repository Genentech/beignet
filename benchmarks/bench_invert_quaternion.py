import random

import torch

import beignet

from ._set_seed import set_seed


class BenchInvertQuaternion:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.invert_quaternion,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.canonical = random.choice([True, False])

    def time_invert_quaternion(self, batch_size, dtype):
        self.func(self.input, self.canonical)

    def peak_memory_invert_quaternion(self, batch_size, dtype):
        self.func(self.input, self.canonical)

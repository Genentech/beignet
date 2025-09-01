import random

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchComposeRotationVector:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.compose_rotation_vector,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.other = torch.randn(batch_size, 3, dtype=dtype)

        self.degrees = random.choice([True, False])

    def time_compose_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.other, self.degrees)

    def peak_memory_compose_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.other, self.degrees)

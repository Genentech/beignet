import random

import torch

import beignet

from ._set_seed import set_seed


class BenchRotationVectorMagnitude:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.rotation_vector_magnitude,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.degrees = random.choice([True, False])

    def time_rotation_vector_magnitude(self, batch_size, dtype):
        self.func(self.input, self.degrees)

    def peak_memory_rotation_vector_magnitude(self, batch_size, dtype):
        self.func(self.input, self.degrees)

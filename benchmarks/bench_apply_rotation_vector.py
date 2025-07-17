import random

import torch

import beignet


class BenchApplyRotationVector:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.apply_rotation_vector,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.rotation = torch.randn(batch_size, 3, dtype=dtype)

        self.degrees = random.choice([True, False])

        self.inverse = random.choice([True, False])

    def time_apply_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.degrees, self.inverse)

    def peak_memory_apply_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.degrees, self.inverse)

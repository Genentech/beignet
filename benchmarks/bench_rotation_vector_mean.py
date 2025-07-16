import random

import torch

import beignet


class RotationVectorMeanBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.rotation_vector_mean,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.weight = random.choice([None, torch.randn(batch_size, dtype=dtype)])

        self.degrees = torch.randn(batch_size, dtype=dtype)

    def time_rotation_vector_mean(self, batch_size, dtype):
        self.func(self.input, self.weight, self.degrees)

    def peak_memory_rotation_vector_mean(self, batch_size, dtype):
        self.func(self.input, self.weight, self.degrees)

import random

import torch

import beignet


class BenchQuaternionToRotationVector:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.quaternion_to_rotation_vector,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 4, dtype=dtype)

        self.degrees = random.choice([True, False])

    def time_quaternion_to_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.degrees)

    def peak_memory_quaternion_to_rotation_vector(self, batch_size, dtype):
        self.func(self.input, self.degrees)

import random

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchRotationVectorToQuaternion:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.rotation_vector_to_quaternion,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.degrees = random.choice([True, False])

        self.canonical = random.choice([True, False])

    def time_rotation_vector_to_quaternion(self, batch_size, dtype):
        self.func(self.input, self.degrees, self.canonical)

    def peak_memory_rotation_vector_to_quaternion(self, batch_size, dtype):
        self.func(self.input, self.degrees, self.canonical)

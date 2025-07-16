import random

import torch

import beignet


class RandomQuaternionBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.random_quaternion,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.size = 10

        self.canonical = random.choice([True, False])

        self.generator = random.choice([None, torch.Generator()])

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

        self.dtype = torch.randn(batch_size, dtype=dtype)

        self.layout = torch.randn(batch_size, dtype=dtype)

        self.device = torch.randn(batch_size, dtype=dtype)

        self.requires_grad = torch.randn(batch_size, dtype=dtype)

        self.pin_memory = torch.randn(batch_size, dtype=dtype)

    def time_random_quaternion(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.canonical,
            self.generator,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
            self.pin_memory,
        )

    def peak_memory_random_quaternion(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.canonical,
            self.generator,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
            self.pin_memory,
        )

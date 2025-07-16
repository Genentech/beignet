import random

import torch

import beignet


class QuaternionIdentityBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.quaternion_identity,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.size = 10

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

        self.dtype = torch.randn(batch_size, dtype=dtype)

        self.layout = torch.randn(batch_size, dtype=dtype)

        self.device = torch.randn(batch_size, dtype=dtype)

        self.requires_grad = torch.randn(batch_size, dtype=dtype)

    def time_quaternion_identity(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
        )

    def peak_memory_quaternion_identity(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
        )

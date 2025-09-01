import random

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchRotationVectorIdentity:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.rotation_vector_identity,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.size = 10

        self.degrees = random.choice([True, False])

        self.out = random.choice([None, torch.empty((self.size, 3), dtype=dtype)])

        self.dtype = dtype

        self.layout = torch.strided

        self.device = torch.device("cpu")

        self.requires_grad = random.choice([True, False])

    def time_rotation_vector_identity(self, batch_size, dtype):
        self.func(
            self.size,
            self.degrees,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def peak_memory_rotation_vector_identity(self, batch_size, dtype):
        self.func(
            self.size,
            self.degrees,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )

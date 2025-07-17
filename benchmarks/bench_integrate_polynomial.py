import random

import torch

import beignet

from ._set_seed import set_seed


class BenchIntegratePolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.integrate_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.order = random.randint(1, 10)

        self.k = torch.randn(batch_size, dtype=dtype)

        self.lower_bound = torch.randn(batch_size, dtype=dtype)

        self.scale = torch.randn(batch_size, dtype=dtype)

        self.dim = random.randint(-1, 2)

    def time_integrate_polynomial(self, batch_size, dtype):
        self.func(
            self.input, self.order, self.k, self.lower_bound, self.scale, self.dim
        )

    def peak_memory_integrate_polynomial(self, batch_size, dtype):
        self.func(
            self.input, self.order, self.k, self.lower_bound, self.scale, self.dim
        )

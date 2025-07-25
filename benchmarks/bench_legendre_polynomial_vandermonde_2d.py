import random

import torch

import beignet

from ._set_seed import set_seed


class BenchLegendrePolynomialVandermonde2D:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.legendre_polynomial_vandermonde_2d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.degree = random.randint(1, 10)

    def time_legendre_polynomial_vandermonde_2d(self, batch_size, dtype):
        self.func(self.x, self.y, self.degree)

    def peak_memory_legendre_polynomial_vandermonde_2d(self, batch_size, dtype):
        self.func(self.x, self.y, self.degree)

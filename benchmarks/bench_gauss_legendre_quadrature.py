import random

import torch

import beignet


class BenchGaussLegendreQuadrature:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.gauss_legendre_quadrature,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.degree = random.randint(1, 10)

    def time_gauss_legendre_quadrature(self, batch_size, dtype):
        self.func(self.input, self.degree)

    def peak_memory_gauss_legendre_quadrature(self, batch_size, dtype):
        self.func(self.input, self.degree)

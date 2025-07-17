import random

import torch

import beignet

from ._set_seed import set_seed


class BenchGaussPhysicistsHermitePolynomialQuadrature:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.gauss_physicists_hermite_polynomial_quadrature,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.degree = random.randint(1, 10)

    def time_gauss_physicists_hermite_polynomial_quadrature(self, batch_size, dtype):
        self.func(self.degree)

    def peak_memory_gauss_physicists_hermite_polynomial_quadrature(
        self, batch_size, dtype
    ):
        self.func(self.degree)

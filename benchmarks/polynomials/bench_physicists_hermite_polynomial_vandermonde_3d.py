import random

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchPhysicistsHermitePolynomialVandermonde3D:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.physicists_hermite_polynomial_vandermonde_3d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.z = torch.randn(batch_size, dtype=dtype)

        self.degree = random.randint(1, 10)

    def time_physicists_hermite_polynomial_vandermonde_3d(self, batch_size, dtype):
        self.func(self.x, self.y, self.z, self.degree)

    def peak_memory_physicists_hermite_polynomial_vandermonde_3d(
        self, batch_size, dtype
    ):
        self.func(self.x, self.y, self.z, self.degree)

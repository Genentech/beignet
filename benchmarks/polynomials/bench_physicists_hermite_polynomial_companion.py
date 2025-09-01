import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchPhysicistsHermitePolynomialCompanion:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.physicists_hermite_polynomial_companion,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

    def time_physicists_hermite_polynomial_companion(self, batch_size, dtype):
        self.func(self.input)

    def peak_memory_physicists_hermite_polynomial_companion(self, batch_size, dtype):
        self.func(self.input)

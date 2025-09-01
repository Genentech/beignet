import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchEvaluatePhysicistsHermitePolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.evaluate_physicists_hermite_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.coefficients = torch.randn(batch_size, 10, dtype=dtype)
        self.tensor = True

    def time_evaluate_physicists_hermite_polynomial(self, batch_size, dtype):
        self.func(self.input, self.coefficients, self.tensor)

    def peak_memory_evaluate_physicists_hermite_polynomial(self, batch_size, dtype):
        self.func(self.input, self.coefficients, self.tensor)

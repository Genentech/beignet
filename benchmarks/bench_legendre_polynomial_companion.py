import torch

import beignet


class LegendrePolynomialCompanionBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.legendre_polynomial_companion,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

    def time_legendre_polynomial_companion(self, batch_size, dtype):
        self.func(self.input)

    def peak_memory_legendre_polynomial_companion(self, batch_size, dtype):
        self.func(self.input)

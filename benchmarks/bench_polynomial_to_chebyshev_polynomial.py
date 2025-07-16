import torch

import beignet


class PolynomialToChebyshevPolynomialBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.polynomial_to_chebyshev_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

    def time_polynomial_to_chebyshev_polynomial(self, batch_size, dtype):
        self.func(self.input)

    def peak_memory_polynomial_to_chebyshev_polynomial(self, batch_size, dtype):
        self.func(self.input)

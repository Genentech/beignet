import random

import torch

import beignet


class FitChebyshevPolynomialBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.fit_chebyshev_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.other = torch.randn(batch_size, dtype=dtype)

        self.degree = random.randint(1, 10)

        self.relative_condition = torch.randn(batch_size, dtype=dtype)

        self.full = random.choice([True, False])

        self.weight = random.choice([None, torch.randn(batch_size, dtype=dtype)])

    def time_fit_chebyshev_polynomial(self, batch_size, dtype):
        self.func(
            self.input,
            self.other,
            self.degree,
            self.relative_condition,
            self.full,
            self.weight,
        )

    def peak_memory_fit_chebyshev_polynomial(self, batch_size, dtype):
        self.func(
            self.input,
            self.other,
            self.degree,
            self.relative_condition,
            self.full,
            self.weight,
        )

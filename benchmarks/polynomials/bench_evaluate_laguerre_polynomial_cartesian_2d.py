import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchEvaluateLaguerrePolynomialCartesian2D:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.evaluate_laguerre_polynomial_cartesian_2d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.c = torch.randn(batch_size, dtype=dtype)

    def time_evaluate_laguerre_polynomial_cartesian_2d(self, batch_size, dtype):
        self.func(self.c, self.x, self.y)

    def peak_memory_evaluate_laguerre_polynomial_cartesian_2d(self, batch_size, dtype):
        self.func(self.c, self.x, self.y)

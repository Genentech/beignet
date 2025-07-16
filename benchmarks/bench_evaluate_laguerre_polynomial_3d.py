import torch

import beignet


class BenchEvaluateLaguerrePolynomial3D:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.evaluate_laguerre_polynomial_3d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.z = torch.randn(batch_size, dtype=dtype)

        self.coefficients = torch.randn(batch_size, 10, dtype=dtype)

    def time_evaluate_laguerre_polynomial_3d(self, batch_size, dtype):
        self.func(self.input)

    def peak_memory_evaluate_laguerre_polynomial_3d(self, batch_size, dtype):
        self.func(self.input)

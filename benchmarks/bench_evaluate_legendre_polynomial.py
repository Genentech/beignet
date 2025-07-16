import torch

import beignet


class EvaluateLegendrePolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.evaluate_legendre_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.coefficients = torch.randn(batch_size, 10, dtype=dtype)
        self.tensor = True

    def time_evaluate_legendre_polynomial(self, batch_size, dtype):
        self.func(self.input, self.coefficients, self.tensor)

    def peak_memory_evaluate_legendre_polynomial(self, batch_size, dtype):
        self.func(self.input, self.coefficients, self.tensor)

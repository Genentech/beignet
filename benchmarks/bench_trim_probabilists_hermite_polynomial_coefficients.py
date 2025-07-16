import torch

import beignet


class BenchTrimProbabilistsHermitePolynomialCoefficients:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.trim_probabilists_hermite_polynomial_coefficients,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.tol = torch.randn(batch_size, dtype=dtype)

    def time_trim_probabilists_hermite_polynomial_coefficients(self, batch_size, dtype):
        self.func(self.input, self.tol)

    def peak_memory_trim_probabilists_hermite_polynomial_coefficients(
        self, batch_size, dtype
    ):
        self.func(self.input, self.tol)

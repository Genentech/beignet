import torch

import beignet


class MultiplyProbabilistsHermitePolynomialByXBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.multiply_probabilists_hermite_polynomial_by_x,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.mode = torch.randn(batch_size, dtype=dtype)

    def time_multiply_probabilists_hermite_polynomial_by_x(self, batch_size, dtype):
        self.func(self.input, self.mode)

    def peak_memory_multiply_probabilists_hermite_polynomial_by_x(
        self, batch_size, dtype
    ):
        self.func(self.input, self.mode)

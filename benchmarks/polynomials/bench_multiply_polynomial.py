import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchMultiplyPolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.multiply_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.other = torch.randn(batch_size, dtype=dtype)

        self.mode = torch.randn(batch_size, dtype=dtype)

    def time_multiply_polynomial(self, batch_size, dtype):
        self.func(self.input, self.other, self.mode)

    def peak_memory_multiply_polynomial(self, batch_size, dtype):
        self.func(self.input, self.other, self.mode)

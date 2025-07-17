import random

import torch

import beignet


class BenchBisect:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.bisect,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, dtype=dtype)

        self.f = lambda x: x**3 - x - 2

        self.a = torch.tensor(-1.0, dtype=dtype)

        self.b = torch.tensor(1.0, dtype=dtype)

        self.rtol = torch.randn(batch_size, dtype=dtype)

        self.atol = torch.randn(batch_size, dtype=dtype)

        self.maxiter = 10

        self.return_solution_info = True

        self.check_bracket = random.choice([True, False])

        self.unroll = 10

    def time_bisect(self, batch_size, dtype):
        self.func(
            self.input,
            self.f,
            self.a,
            self.b,
            self.rtol,
            self.atol,
            self.maxiter,
            self.return_solution_info,
            self.check_bracket,
            self.unroll,
        )

    def peak_memory_bisect(self, batch_size, dtype):
        self.func(
            self.input,
            self.f,
            self.a,
            self.b,
            self.rtol,
            self.atol,
            self.maxiter,
            self.return_solution_info,
            self.check_bracket,
            self.unroll,
        )

import random

import torch

import beignet


class DawsonIntegralFBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.special.dawson_integral_f,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, dtype=dtype)

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

    def time_dawson_integral_f(self, batch_size, dtype):
        self.func(self.input, self.out)

    def peak_memory_dawson_integral_f(self, batch_size, dtype):
        self.func(self.input, self.out)

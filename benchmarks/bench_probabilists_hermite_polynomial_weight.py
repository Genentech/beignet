import torch

import beignet

from ._set_seed import set_seed


class BenchProbabilistsHermitePolynomialWeight:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.probabilists_hermite_polynomial_weight,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

    def time_probabilists_hermite_polynomial_weight(self, batch_size, dtype):
        self.func(self.x)

    def peak_memory_probabilists_hermite_polynomial_weight(self, batch_size, dtype):
        self.func(self.x)

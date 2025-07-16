import random

import torch

import beignet


class IdentityMatrixBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.identity_matrix,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.d = random.randint(1, 10)

        self.size = torch.randn(batch_size, dtype=dtype)

        self.dtype = torch.randn(batch_size, dtype=dtype)

        self.device = torch.randn(batch_size, dtype=dtype)

    def time_identity_matrix(self, batch_size, dtype):
        self.func(self.input, self.d, self.size, self.dtype, self.device)

    def peak_memory_identity_matrix(self, batch_size, dtype):
        self.func(self.input, self.d, self.size, self.dtype, self.device)

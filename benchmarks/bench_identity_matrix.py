import random

import torch

import beignet

from ._set_seed import set_seed


class BenchIdentityMatrix:
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
        set_seed()

        self.d = random.randint(1, 10)

        self.size = ()

        self.dtype = dtype

        self.device = torch.device("cpu")

    def time_identity_matrix(self, batch_size, dtype):
        self.func(self.d, self.size, self.dtype, self.device)

    def peak_memory_identity_matrix(self, batch_size, dtype):
        self.func(self.d, self.size, self.dtype, self.device)

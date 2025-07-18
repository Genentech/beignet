import torch

import beignet

from ._set_seed import set_seed


class BenchLennardJonesPotential:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.lennard_jones_potential,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.sigma = torch.rand(batch_size, dtype=dtype) * 10.0 + 0.1

        self.epsilon = torch.rand(batch_size, dtype=dtype) * 10.0 + 0.1

    def time_lennard_jones_potential(self, batch_size, dtype):
        self.func(self.input, self.sigma, self.epsilon)

    def peak_memory_lennard_jones_potential(self, batch_size, dtype):
        self.func(self.input, self.sigma, self.epsilon)

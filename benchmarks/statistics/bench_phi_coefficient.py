import torch

import beignet

from ._set_seed import set_seed


class BenchPhiCoefficient:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.phi_coefficient,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.chi_square_values = (
            torch.tensor([1.0, 4.0, 9.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_phi_coefficient(self, batch_size, dtype):
        return self.func(self.chi_square_values, self.sample_sizes)

    def peakmem_phi_coefficient(self, batch_size, dtype):
        return beignet.statistics.phi_coefficient(
            self.chi_square_values, self.sample_sizes
        )

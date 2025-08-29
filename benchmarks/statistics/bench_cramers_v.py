import torch

import beignet

from ._set_seed import set_seed


class BenchCramersV:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.cramers_v,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.chi_square_values = (
            torch.tensor([1.0, 5.5, 12.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.min_dims = (
            torch.tensor([1, 2, 3], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_cramers_v(self, batch_size, dtype):
        return self.func(self.chi_square_values, self.sample_sizes, self.min_dims)

    def peakmem_cramers_v(self, batch_size, dtype):
        return beignet.statistics.cramers_v(
            self.chi_square_values,
            self.sample_sizes,
            self.min_dims,
        )

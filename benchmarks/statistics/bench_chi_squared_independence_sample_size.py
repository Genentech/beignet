import torch

import beignet

from ._set_seed import set_seed


class BenchChiSquareIndependenceSampleSize:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.chi_squared_independence_sample_size,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.rows = torch.randint(2, 5, (batch_size,), dtype=dtype)
        self.cols = torch.randint(2, 5, (batch_size,), dtype=dtype)

    def time_chisquare_independence_sample_size(self, batch_size, dtype):
        return self.func(self.effect_size, self.rows, self.cols, power=0.8, alpha=0.05)

    def peakmem_chisquare_independence_sample_size(self, batch_size, dtype):
        return beignet.statistics.chi_squared_independence_sample_size(
            self.effect_size,
            self.rows,
            self.cols,
            power=0.8,
            alpha=0.05,
        )

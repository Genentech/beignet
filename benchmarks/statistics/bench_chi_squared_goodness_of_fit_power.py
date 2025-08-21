import torch

import beignet

from ._set_seed import set_seed


class BenchChiSquareGoodnessOfFitPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.chi_squared_goodness_of_fit_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.sample_size = torch.randint(20, 200, (batch_size,), dtype=dtype)
        self.df = torch.randint(1, 10, (batch_size,), dtype=dtype)

    def time_chisquare_gof_power(self, batch_size, dtype):
        return self.func(self.effect_size, self.sample_size, self.df, alpha=0.05)

    def peakmem_chisquare_gof_power(self, batch_size, dtype):
        return beignet.statistics.chi_squared_goodness_of_fit_power(
            self.effect_size, self.sample_size, self.df, alpha=0.05
        )

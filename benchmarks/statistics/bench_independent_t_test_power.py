import torch

import beignet

from ._set_seed import set_seed


class BenchIndependentTTestPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.independent_t_test_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.nobs1_values = (
            torch.tensor([20, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_independent_t_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.nobs1_values,
            alpha=0.05,
            alternative="two-sided",
            ratio=self.ratio_values,
        )

    def peak_memory_independent_t_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.nobs1_values,
            alpha=0.05,
            alternative="two-sided",
            ratio=self.ratio_values,
        )

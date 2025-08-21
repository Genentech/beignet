import torch

import beignet

from ._set_seed import set_seed


class BenchAnovaPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.anova_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([60, 120, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_anova_power_alpha_005(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.05
        )

    def time_anova_power_alpha_001(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.01
        )

    def peakmem_anova_power(self, batch_size, dtype):
        return beignet.statistics.anova_power(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.05
        )

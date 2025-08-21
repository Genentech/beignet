import torch

import beignet

from ._set_seed import set_seed


class BenchAnovaSampleSize:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.anova_sample_size,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_anova_sample_size_power_08(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.05
        )

    def time_anova_sample_size_power_09(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, self.k_values, power=0.9, alpha=0.05
        )

    def time_anova_sample_size_alpha_001(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.01
        )

    def peakmem_anova_sample_size(self, batch_size, dtype):
        return beignet.statistics.anova_sample_size(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.05
        )

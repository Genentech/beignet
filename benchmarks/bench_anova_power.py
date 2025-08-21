import os

import torch

import beignet


class TimeAnovaPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([60, 120, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_anova_power = torch.compile(beignet.anova_power, fullgraph=True)

    def time_anova_power_alpha_005(self, batch_size, dtype):
        return self.compiled_anova_power(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.05
        )

    def time_anova_power_alpha_001(self, batch_size, dtype):
        return self.compiled_anova_power(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.01
        )


class PeakMemoryAnovaPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([60, 120, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_anova_power(self, batch_size, dtype):
        return beignet.anova_power(
            self.effect_sizes, self.sample_sizes, self.k_values, alpha=0.05
        )

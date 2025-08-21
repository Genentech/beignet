import os

import torch

import beignet


class TimeAnovaSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_anova_sample_size = torch.compile(
            beignet.anova_sample_size, fullgraph=True
        )

    def time_anova_sample_size_power_08(self, batch_size, dtype):
        return self.compiled_anova_sample_size(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.05
        )

    def time_anova_sample_size_power_09(self, batch_size, dtype):
        return self.compiled_anova_sample_size(
            self.effect_sizes, self.k_values, power=0.9, alpha=0.05
        )

    def time_anova_sample_size_alpha_001(self, batch_size, dtype):
        return self.compiled_anova_sample_size(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.01
        )


class PeakMemoryAnovaSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.1, 0.25, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.k_values = (
            torch.tensor([3, 4, 5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_anova_sample_size(self, batch_size, dtype):
        return beignet.anova_sample_size(
            self.effect_sizes, self.k_values, power=0.8, alpha=0.05
        )

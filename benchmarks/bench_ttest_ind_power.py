import os

import torch

import beignet


class TimeTTestIndPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.nobs1_values = (
            torch.tensor([20, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_ttest_ind_power = torch.compile(
            beignet.ttest_ind_power, fullgraph=True
        )

    def time_ttest_ind_power_two_sided(self, batch_size, dtype):
        return self.compiled_ttest_ind_power(
            self.effect_sizes,
            self.nobs1_values,
            self.ratio_values,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_ttest_ind_power_larger(self, batch_size, dtype):
        return self.compiled_ttest_ind_power(
            self.effect_sizes,
            self.nobs1_values,
            self.ratio_values,
            alpha=0.05,
            alternative="larger",
        )

    def time_ttest_ind_power_balanced(self, batch_size, dtype):
        ratio_ones = torch.ones_like(self.effect_sizes)
        return self.compiled_ttest_ind_power(
            self.effect_sizes,
            self.nobs1_values,
            ratio_ones,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_ttest_ind_power_alpha_001(self, batch_size, dtype):
        return self.compiled_ttest_ind_power(
            self.effect_sizes,
            self.nobs1_values,
            self.ratio_values,
            alpha=0.01,
            alternative="two-sided",
        )


class PeakMemoryTTestIndPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.nobs1_values = (
            torch.tensor([20, 30, 50], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_ttest_ind_power(self, batch_size, dtype):
        return beignet.ttest_ind_power(
            self.effect_sizes,
            self.nobs1_values,
            self.ratio_values,
            alpha=0.05,
            alternative="two-sided",
        )

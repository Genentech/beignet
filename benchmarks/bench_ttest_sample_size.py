import os

import torch

import beignet


class TimeTTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_ttest_sample_size = torch.compile(
            beignet.ttest_sample_size, fullgraph=True
        )

    def time_ttest_sample_size_power_08(self, batch_size, dtype):
        return self.compiled_ttest_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )

    def time_ttest_sample_size_power_09(self, batch_size, dtype):
        return self.compiled_ttest_sample_size(
            self.effect_sizes, power=0.9, alpha=0.05, alternative="two-sided"
        )

    def time_ttest_sample_size_one_sided(self, batch_size, dtype):
        return self.compiled_ttest_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="one-sided"
        )

    def time_ttest_sample_size_alpha_001(self, batch_size, dtype):
        return self.compiled_ttest_sample_size(
            self.effect_sizes, power=0.8, alpha=0.01, alternative="two-sided"
        )


class PeakMemoryTTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_ttest_sample_size(self, batch_size, dtype):
        return beignet.ttest_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )

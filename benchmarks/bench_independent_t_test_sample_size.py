import os

import torch

import beignet


class TimeIndependentTTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_independent_t_test_sample_size = torch.compile(
            beignet.independent_t_test_sample_size, fullgraph=True
        )

    def time_independent_t_test_sample_size_power_08(self, batch_size, dtype):
        return self.compiled_independent_t_test_sample_size(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_independent_t_test_sample_size_power_09(self, batch_size, dtype):
        return self.compiled_independent_t_test_sample_size(
            self.effect_sizes,
            self.ratio_values,
            power=0.9,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_independent_t_test_sample_size_balanced(self, batch_size, dtype):
        ratio_ones = torch.ones_like(self.effect_sizes)
        return self.compiled_independent_t_test_sample_size(
            self.effect_sizes,
            ratio_ones,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_independent_t_test_sample_size_alpha_001(self, batch_size, dtype):
        return self.compiled_independent_t_test_sample_size(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.01,
            alternative="two-sided",
        )

    def time_independent_t_test_sample_size_one_sided(self, batch_size, dtype):
        return self.compiled_independent_t_test_sample_size(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.05,
            alternative="larger",
        )


class PeakMemoryIndependentTTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_independent_t_test_sample_size(self, batch_size, dtype):
        return beignet.independent_t_test_sample_size(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

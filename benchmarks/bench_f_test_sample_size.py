import os

import torch

import beignet


class TimeFTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.df1_values = (
            torch.tensor([2.0, 5.0, 10.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_f_test_sample_size = torch.compile(
            beignet.f_test_sample_size, fullgraph=True
        )

    def time_f_test_sample_size_power_08(self, batch_size, dtype):
        return self.compiled_f_test_sample_size(
            self.effect_sizes,
            self.df1_values,
            power=0.8,
            alpha=0.05,
        )

    def time_f_test_sample_size_power_09(self, batch_size, dtype):
        return self.compiled_f_test_sample_size(
            self.effect_sizes,
            self.df1_values,
            power=0.9,
            alpha=0.05,
        )

    def time_f_test_sample_size_alpha_01(self, batch_size, dtype):
        return self.compiled_f_test_sample_size(
            self.effect_sizes,
            self.df1_values,
            power=0.8,
            alpha=0.01,
        )

    def time_f_test_sample_size_small_df(self, batch_size, dtype):
        small_df1 = torch.full_like(self.effect_sizes, 1.0)
        return self.compiled_f_test_sample_size(
            self.effect_sizes,
            small_df1,
            power=0.8,
            alpha=0.05,
        )

    def time_f_test_sample_size_large_df(self, batch_size, dtype):
        large_df1 = torch.full_like(self.effect_sizes, 20.0)
        return self.compiled_f_test_sample_size(
            self.effect_sizes,
            large_df1,
            power=0.8,
            alpha=0.05,
        )


class PeakMemoryFTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.df1_values = (
            torch.tensor([2.0, 5.0, 10.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_f_test_sample_size(self, batch_size, dtype):
        return beignet.f_test_sample_size(
            self.effect_sizes,
            self.df1_values,
            power=0.8,
            alpha=0.05,
        )

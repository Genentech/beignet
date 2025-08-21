import os

import torch

import beignet


class TimeZTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_z_test_sample_size = torch.compile(
            beignet.z_test_sample_size, fullgraph=True
        )

    def time_z_test_sample_size_power_08(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_z_test_sample_size_power_09(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.9,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_z_test_sample_size_power_07(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.7,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_z_test_sample_size_alpha_01(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.8,
            alpha=0.01,
            alternative="two-sided",
        )

    def time_z_test_sample_size_greater(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.8,
            alpha=0.05,
            alternative="greater",
        )

    def time_z_test_sample_size_less(self, batch_size, dtype):
        return self.compiled_z_test_sample_size(
            self.effect_sizes,
            power=0.8,
            alpha=0.05,
            alternative="less",
        )


class PeakMemoryZTestSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_z_test_sample_size(self, batch_size, dtype):
        return beignet.z_test_sample_size(
            self.effect_sizes,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

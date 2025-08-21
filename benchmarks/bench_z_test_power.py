import os

import torch

import beignet


class TimeZTestPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_z_test_power = torch.compile(beignet.z_test_power, fullgraph=True)

    def time_z_test_power_two_sided(self, batch_size, dtype):
        return self.compiled_z_test_power(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_z_test_power_greater(self, batch_size, dtype):
        return self.compiled_z_test_power(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="greater",
        )

    def time_z_test_power_less(self, batch_size, dtype):
        return self.compiled_z_test_power(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="less",
        )

    def time_z_test_power_alpha_01(self, batch_size, dtype):
        return self.compiled_z_test_power(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.01,
            alternative="two-sided",
        )


class PeakMemoryZTestPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_z_test_power(self, batch_size, dtype):
        return beignet.z_test_power(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

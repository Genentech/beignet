import os

import torch

import beignet


class TimeProportionPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.p0_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p1_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_proportion_power = torch.compile(
            beignet.proportion_power, fullgraph=True
        )

    def time_proportion_power_two_sided(self, batch_size, dtype):
        return self.compiled_proportion_power(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_proportion_power_greater(self, batch_size, dtype):
        return self.compiled_proportion_power(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="greater",
        )

    def time_proportion_power_less(self, batch_size, dtype):
        return self.compiled_proportion_power(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="less",
        )


class PeakMemoryProportionPower:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.p0_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p1_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_proportion_power(self, batch_size, dtype):
        return beignet.proportion_power(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

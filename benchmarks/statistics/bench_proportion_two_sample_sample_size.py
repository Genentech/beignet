import os

import torch

import beignet


class TimeProportionTwoSampleSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.p1_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p2_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_proportion_two_sample_sample_size = torch.compile(
            beignet.proportion_two_sample_sample_size, fullgraph=True
        )

    def time_proportion_two_sample_sample_size_two_sided(self, batch_size, dtype):
        return self.compiled_proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_greater(self, batch_size, dtype):
        return self.compiled_proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="greater",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_less(self, batch_size, dtype):
        return self.compiled_proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="less",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_unequal(self, batch_size, dtype):
        return self.compiled_proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=2.0,
        )


class PeakMemoryProportionTwoSampleSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.p1_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p2_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_proportion_two_sample_sample_size(self, batch_size, dtype):
        return beignet.proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=1.0,
        )

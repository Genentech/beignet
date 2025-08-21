import os

import torch

import beignet


class TimeCorrelationSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.r_values = (
            torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_correlation_sample_size = torch.compile(
            beignet.correlation_sample_size, fullgraph=True
        )

    def time_correlation_sample_size_two_sided(self, batch_size, dtype):
        return self.compiled_correlation_sample_size(
            self.r_values, power=0.8, alpha=0.05, alternative="two-sided"
        )

    def time_correlation_sample_size_greater(self, batch_size, dtype):
        return self.compiled_correlation_sample_size(
            self.r_values, power=0.8, alpha=0.05, alternative="greater"
        )

    def time_correlation_sample_size_less(self, batch_size, dtype):
        return self.compiled_correlation_sample_size(
            self.r_values, power=0.8, alpha=0.05, alternative="less"
        )


class PeakMemoryCorrelationSampleSize:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.r_values = (
            torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_correlation_sample_size(self, batch_size, dtype):
        return beignet.correlation_sample_size(
            self.r_values, power=0.8, alpha=0.05, alternative="two-sided"
        )

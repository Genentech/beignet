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
        self.compiled_t_test_sample_size = torch.compile(
            beignet.t_test_sample_size, fullgraph=True
        )

    def time_t_test_sample_size_two_sided(self, batch_size, dtype):
        return self.compiled_t_test_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )

    def time_t_test_sample_size_greater(self, batch_size, dtype):
        return self.compiled_t_test_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="greater"
        )

    def time_t_test_sample_size_less(self, batch_size, dtype):
        return self.compiled_t_test_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="less"
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

    def peakmem_t_test_sample_size(self, batch_size, dtype):
        return beignet.t_test_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )

"""Benchmarks for chi-square independence sample size."""

import os

import torch

from beignet import chisquare_independence_sample_size

# Set random seed for reproducible benchmarks
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeChiSquareIndependenceSampleSize:
    """Benchmark chi-square independence sample size computation time."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Set up benchmark parameters."""
        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.rows = torch.randint(2, 5, (batch_size,), dtype=dtype)
        self.cols = torch.randint(2, 5, (batch_size,), dtype=dtype)

        # Compile for optimal performance
        self.compiled_func = torch.compile(
            chisquare_independence_sample_size, fullgraph=True
        )

    def time_chisquare_independence_sample_size(self, batch_size, dtype):
        """Benchmark chi-square independence sample size computation."""
        return self.compiled_func(
            self.effect_size, self.rows, self.cols, power=0.8, alpha=0.05
        )


class PeakMemoryChiSquareIndependenceSampleSize:
    """Benchmark chi-square independence sample size memory usage."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Set up benchmark parameters."""
        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.rows = torch.randint(2, 5, (batch_size,), dtype=dtype)
        self.cols = torch.randint(2, 5, (batch_size,), dtype=dtype)

        # Compile for optimal performance
        self.compiled_func = torch.compile(
            chisquare_independence_sample_size, fullgraph=True
        )

    def peakmem_chisquare_independence_sample_size(self, batch_size, dtype):
        """Benchmark chi-square independence sample size memory usage."""
        return self.compiled_func(
            self.effect_size, self.rows, self.cols, power=0.8, alpha=0.05
        )

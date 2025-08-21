"""Benchmarks for chi-square goodness-of-fit sample size."""

import os

import torch

from beignet import chisquare_gof_sample_size

# Set random seed for reproducible benchmarks
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeChiSquareGoodnessOfFitSampleSize:
    """Benchmark chi-square goodness-of-fit sample size computation time."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Set up benchmark parameters."""
        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.df = torch.randint(1, 10, (batch_size,), dtype=dtype)

        # Compile for optimal performance
        self.compiled_func = torch.compile(chisquare_gof_sample_size, fullgraph=True)

    def time_chisquare_gof_sample_size(self, batch_size, dtype):
        """Benchmark chi-square goodness-of-fit sample size computation."""
        return self.compiled_func(self.effect_size, self.df, power=0.8, alpha=0.05)


class PeakMemoryChiSquareGoodnessOfFitSampleSize:
    """Benchmark chi-square goodness-of-fit sample size memory usage."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Set up benchmark parameters."""
        self.effect_size = torch.rand(batch_size, dtype=dtype) * 0.5 + 0.1
        self.df = torch.randint(1, 10, (batch_size,), dtype=dtype)

        # Compile for optimal performance
        self.compiled_func = torch.compile(chisquare_gof_sample_size, fullgraph=True)

    def peakmem_chisquare_gof_sample_size(self, batch_size, dtype):
        """Benchmark chi-square goodness-of-fit sample size memory usage."""
        return self.compiled_func(self.effect_size, self.df, power=0.8, alpha=0.05)

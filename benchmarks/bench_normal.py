"""Benchmark for Normal distribution with icdf."""

import torch

import beignet.distributions


class TimeNormal:
    """Benchmark Normal distribution icdf method."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup benchmark data."""
        # Standard normal distribution
        self.loc = torch.zeros(batch_size, dtype=dtype)
        self.scale = torch.ones(batch_size, dtype=dtype)
        self.dist = beignet.distributions.Normal(self.loc, self.scale)

        # Test probabilities
        self.probabilities = torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975], dtype=dtype)

        # Compiled version
        self.compiled_dist = beignet.distributions.Normal(self.loc[:1], self.scale[:1])
        self.compiled_icdf = torch.compile(self.compiled_dist.icdf, fullgraph=True)

    def time_normal_icdf(self, batch_size, dtype):
        """Benchmark Normal distribution icdf."""
        return self.dist.icdf(self.probabilities)

    def time_normal_icdf_compiled(self, batch_size, dtype):
        """Benchmark compiled Normal distribution icdf."""
        return self.compiled_icdf(self.probabilities)


class PeakMemoryNormal:
    """Benchmark Normal distribution memory usage."""

    params = ([100, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup benchmark data."""
        self.loc = torch.zeros(batch_size, dtype=dtype)
        self.scale = torch.ones(batch_size, dtype=dtype)
        self.dist = beignet.distributions.Normal(self.loc, self.scale)
        self.probabilities = torch.linspace(0.01, 0.99, 100, dtype=dtype)

    def peakmem_normal_icdf(self, batch_size, dtype):
        """Benchmark Normal distribution icdf memory usage."""
        return self.dist.icdf(self.probabilities)

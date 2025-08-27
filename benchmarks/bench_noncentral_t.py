"""Benchmark for NonCentralT distribution with icdf."""

import torch

import beignet.distributions


class TimeNonCentralT:
    """Benchmark NonCentralT distribution icdf method."""

    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup benchmark data."""
        # Non-central t-distribution parameters
        self.df = torch.full((batch_size,), 10.0, dtype=dtype)
        self.nc = torch.full((batch_size,), 1.5, dtype=dtype)
        self.dist = beignet.distributions.NonCentralT(self.df, self.nc)

        # Test probabilities
        self.probabilities = torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975], dtype=dtype)

        # Compiled version
        self.compiled_dist = beignet.distributions.NonCentralT(self.df[:1], self.nc[:1])
        self.compiled_icdf = torch.compile(self.compiled_dist.icdf, fullgraph=True)

    def time_noncentral_t_icdf(self, batch_size, dtype):
        """Benchmark NonCentralT distribution icdf."""
        return self.dist.icdf(self.probabilities)

    def time_noncentral_t_icdf_compiled(self, batch_size, dtype):
        """Benchmark compiled NonCentralT distribution icdf."""
        return self.compiled_icdf(self.probabilities)


class PeakMemoryNonCentralT:
    """Benchmark NonCentralT distribution memory usage."""

    params = ([100, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup benchmark data."""
        self.df = torch.full((batch_size,), 8.0, dtype=dtype)
        self.nc = torch.full((batch_size,), 2.0, dtype=dtype)
        self.dist = beignet.distributions.NonCentralT(self.df, self.nc)
        self.probabilities = torch.linspace(0.01, 0.99, 100, dtype=dtype)

    def peakmem_noncentral_t_icdf(self, batch_size, dtype):
        """Benchmark NonCentralT distribution icdf memory usage."""
        return self.dist.icdf(self.probabilities)

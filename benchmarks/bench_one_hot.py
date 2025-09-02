import torch

from beignet import one_hot


class TimeOneHot:
    """Benchmark one_hot function."""

    params = ([1, 10, 100], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.x = torch.randn(batch_size, 32, dtype=dtype)
        self.v_bins = torch.linspace(-3.0, 3.0, 16, dtype=dtype)

    def time_one_hot(self, batch_size, dtype):
        """Benchmark one_hot function."""
        return one_hot(self.x, self.v_bins)


class PeakMemoryOneHot:
    """Benchmark memory usage of one_hot function."""

    params = ([1, 10], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.x = torch.randn(batch_size, 32, dtype=dtype)
        self.v_bins = torch.linspace(-3.0, 3.0, 16, dtype=dtype)

    def peakmem_one_hot(self, batch_size, dtype):
        """Benchmark memory usage of one_hot function."""
        return one_hot(self.x, self.v_bins)

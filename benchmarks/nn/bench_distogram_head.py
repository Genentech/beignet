import torch

from beignet.nn import AlphaFold3Distogram


class TimeAlphaFold3Distogram:
    """Benchmark AlphaFold3Distogram module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3Distogram(
            c_z=128,
            n_bins=64,
        )

        n_tokens = 64

        # Create test inputs
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 128, dtype=dtype)

    def time_distogram_head(self, batch_size, dtype):
        """Benchmark AlphaFold3Distogram forward pass."""
        return self.module(self.z_ij)


class PeakMemoryAlphaFold3Distogram:
    """Benchmark memory usage of AlphaFold3Distogram module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3Distogram(
            c_z=64,  # Smaller for memory test
            n_bins=32,
        )

        n_tokens = 32  # Smaller for memory test

        # Create test inputs
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 64, dtype=dtype)

    def peakmem_distogram_head(self, batch_size, dtype):
        """Benchmark memory usage of AlphaFold3Distogram forward pass."""
        return self.module(self.z_ij)

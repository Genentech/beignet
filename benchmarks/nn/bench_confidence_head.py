import torch

from beignet.nn import AlphaFold3Confidence


class TimeAlphaFold3Confidence:
    """Benchmark AlphaFold3Confidence module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3Confidence(
            c_s=32,
            c_z=16,
            n_blocks=2,  # Smaller for benchmark
        )

        n_tokens = 32

        # Create input dictionaries
        self.s_inputs = {
            "token_single_initial_repr": torch.randn(
                batch_size, n_tokens, 32, dtype=dtype
            ),
        }

        self.s_i = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)
        self.x_pred = torch.randn(batch_size, n_tokens, 3, dtype=dtype)

    def time_confidence_head(self, batch_size, dtype):
        """Benchmark AlphaFold3Confidence forward pass."""
        return self.module(self.s_inputs, self.s_i, self.z_ij, self.x_pred)


class PeakMemoryAlphaFold3Confidence:
    """Benchmark memory usage of AlphaFold3Confidence module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3Confidence(
            c_s=32,
            c_z=16,
            n_blocks=1,  # Even smaller for memory test
        )

        n_tokens = 16  # Smaller for memory test

        # Create input dictionaries
        self.s_inputs = {
            "token_single_initial_repr": torch.randn(
                batch_size, n_tokens, 32, dtype=dtype
            ),
        }

        self.s_i = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)
        self.x_pred = torch.randn(batch_size, n_tokens, 3, dtype=dtype)

    def peakmem_confidence_head(self, batch_size, dtype):
        """Benchmark memory usage of AlphaFold3Confidence forward pass."""
        return self.module(self.s_inputs, self.s_i, self.z_ij, self.x_pred)

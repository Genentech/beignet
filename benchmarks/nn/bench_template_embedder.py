import torch

from beignet.nn import AlphaFold3TemplateEmbedder


class TimeAlphaFold3TemplateEmbedder:
    """Benchmark AlphaFold3TemplateEmbedder module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3TemplateEmbedder(
            c_z=128,
            c_template=64,
            n_head=4,
        )

        n_tokens = 32

        # Create test inputs
        self.f_star = {
            "template_features": torch.randn(
                batch_size, n_tokens, n_tokens, 64, dtype=dtype
            )
        }
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 128, dtype=dtype)

    def time_template_embedder(self, batch_size, dtype):
        """Benchmark AlphaFold3TemplateEmbedder forward pass."""
        return self.module(self.f_star, self.z_ij)


class PeakMemoryAlphaFold3TemplateEmbedder:
    """Benchmark memory usage of AlphaFold3TemplateEmbedder module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3TemplateEmbedder(
            c_z=64,  # Smaller for memory test
            c_template=32,
            n_head=4,
        )

        n_tokens = 16  # Smaller for memory test

        # Create test inputs
        self.f_star = {
            "template_features": torch.randn(
                batch_size, n_tokens, n_tokens, 32, dtype=dtype
            )
        }
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 64, dtype=dtype)

    def peakmem_template_embedder(self, batch_size, dtype):
        """Benchmark memory usage of AlphaFold3TemplateEmbedder forward pass."""
        return self.module(self.f_star, self.z_ij)

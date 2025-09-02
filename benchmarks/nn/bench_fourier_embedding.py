import os

import torch

from beignet.nn import _FourierEmbedding

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeFourierEmbedding:
    """Benchmark FourierEmbedding (Algorithm 22)."""

    params = (
        [1, 4, 16],
        [10, 20, 50],
        [64, 128, 256],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "seq_len", "c", "dtype"]

    def setup(self, batch_size, seq_len, c, dtype):
        device = torch.device("cpu")

        # Create module and compile for optimal performance
        module = _FourierEmbedding(c=c).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data - both input shapes (..., 1) and (...)
        self.t_with_dim = torch.randn(
            batch_size, seq_len, 1, dtype=dtype, device=device
        )
        self.t_without_dim = torch.randn(
            batch_size, seq_len, dtype=dtype, device=device
        )

    def time_fourier_embedding_with_dim(self, batch_size, seq_len, c, dtype):
        """Benchmark forward pass of FourierEmbedding with shape (..., 1)."""
        return self.module(self.t_with_dim)

    def time_fourier_embedding_without_dim(self, batch_size, seq_len, c, dtype):
        """Benchmark forward pass of FourierEmbedding with shape (...)."""
        return self.module(self.t_without_dim)

    def peakmem_fourier_embedding_with_dim(self, batch_size, seq_len, c, dtype):
        """Benchmark peak memory usage of FourierEmbedding with shape (..., 1)."""
        return self.module(self.t_with_dim)

    def peakmem_fourier_embedding_without_dim(self, batch_size, seq_len, c, dtype):
        """Benchmark peak memory usage of FourierEmbedding with shape (...)."""
        return self.module(self.t_without_dim)


class TimeFourierEmbeddingLarge:
    """Benchmark large-scale FourierEmbedding configurations."""

    params = ([1, 4], [100, 200], [512, 1024], [torch.float32])
    param_names = ["batch_size", "seq_len", "c", "dtype"]

    def setup(self, batch_size, seq_len, c, dtype):
        device = torch.device("cpu")

        # Create module and compile for optimal performance
        module = _FourierEmbedding(c=c).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.t = torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device)

    def time_fourier_embedding_large(self, batch_size, seq_len, c, dtype):
        """Benchmark forward pass of large FourierEmbedding."""
        return self.module(self.t)

    def peakmem_fourier_embedding_large(self, batch_size, seq_len, c, dtype):
        """Benchmark peak memory usage of large FourierEmbedding."""
        return self.module(self.t)


class TimeFourierEmbeddingScaling:
    """Benchmark FourierEmbedding scaling with sequence length."""

    params = ([10, 50, 100, 200], [128], [torch.float32])
    param_names = ["seq_len", "c", "dtype"]

    def setup(self, seq_len, c, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module and compile for optimal performance
        module = _FourierEmbedding(c=c).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.t = torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device)

    def time_fourier_embedding_scaling(self, seq_len, c, dtype):
        """Benchmark FourierEmbedding scaling with sequence length."""
        return self.module(self.t)

    def peakmem_fourier_embedding_scaling(self, seq_len, c, dtype):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.t)


class TimeFourierEmbeddingGradients:
    """Benchmark FourierEmbedding gradient computation."""

    params = ([1, 2], [20, 50], [64, 128], [torch.float32])
    param_names = ["batch_size", "seq_len", "c", "dtype"]

    def setup(self, batch_size, seq_len, c, dtype):
        device = torch.device("cpu")

        # Create module (don't compile for gradient benchmarks)
        self.module = _FourierEmbedding(c=c).to(device).to(dtype)

        # Generate test data
        self.t = torch.randn(
            batch_size, seq_len, 1, dtype=dtype, device=device, requires_grad=True
        )

    def time_fourier_embedding_backward(self, batch_size, seq_len, c, dtype):
        """Benchmark backward pass of FourierEmbedding."""
        # Zero gradients
        if self.t.grad is not None:
            self.t.grad.zero_()

        # Forward pass
        embeddings = self.module(self.t)
        loss = embeddings.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_fourier_embedding_backward(self, batch_size, seq_len, c, dtype):
        """Benchmark peak memory usage of FourierEmbedding backward pass."""
        # Zero gradients
        if self.t.grad is not None:
            self.t.grad.zero_()

        # Forward pass
        embeddings = self.module(self.t)
        loss = embeddings.sum()

        # Backward pass
        loss.backward()

        return loss


class TimeFourierEmbeddingDifferentValues:
    """Benchmark FourierEmbedding with different input value ranges."""

    params = (["small", "medium", "large"], [64], [torch.float32])
    param_names = ["value_range", "c", "dtype"]

    def setup(self, value_range, c, dtype):
        device = torch.device("cpu")
        batch_size, seq_len = 4, 20

        # Create module and compile for optimal performance
        module = _FourierEmbedding(c=c).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data with different value ranges
        if value_range == "small":
            self.t = (
                torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device) * 1e-3
            )
        elif value_range == "medium":
            self.t = torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device)
        elif value_range == "large":
            self.t = (
                torch.randn(batch_size, seq_len, 1, dtype=dtype, device=device) * 1000
            )

    def time_fourier_embedding_values(self, value_range, c, dtype):
        """Benchmark FourierEmbedding with different value ranges."""
        return self.module(self.t)

    def peakmem_fourier_embedding_values(self, value_range, c, dtype):
        """Benchmark peak memory with different value ranges."""
        return self.module(self.t)

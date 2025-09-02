import os

import torch

from beignet.nn import AdaptiveLayerNorm, ConditionedTransitionBlock

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeAdaptiveLayerNorm:
    """Benchmark AdaptiveLayerNorm (Algorithm 26)."""

    params = (
        [1, 4, 16],
        [64, 128, 256],
        [64, 128, 384],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "c", "c_s", "dtype"]

    def setup(self, batch_size, c, c_s, dtype):
        device = torch.device("cpu")
        self.seq_len = 20

        # Create module and compile for optimal performance
        module = AdaptiveLayerNorm(c=c, c_s=c_s).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, self.seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, self.seq_len, c_s, dtype=dtype, device=device)

    def time_adaptive_layer_norm(self, batch_size, c, c_s, dtype):
        """Benchmark forward pass of AdaptiveLayerNorm."""
        return self.module(self.a, self.s)

    def peakmem_adaptive_layer_norm(self, batch_size, c, c_s, dtype):
        """Benchmark peak memory usage of AdaptiveLayerNorm."""
        return self.module(self.a, self.s)


class TimeConditionedTransitionBlock:
    """Benchmark ConditionedTransitionBlock (Algorithm 25)."""

    params = (
        [1, 4, 16],
        [64, 128, 256],
        [64, 128, 384],
        [2, 4, 8],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "c", "c_s", "n", "dtype"]

    def setup(self, batch_size, c, c_s, n, dtype):
        device = torch.device("cpu")
        self.seq_len = 15

        # Create module and compile for optimal performance
        module = ConditionedTransitionBlock(c=c, c_s=c_s, n=n).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, self.seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, self.seq_len, c_s, dtype=dtype, device=device)

    def time_conditioned_transition_block(self, batch_size, c, c_s, n, dtype):
        """Benchmark forward pass of ConditionedTransitionBlock."""
        return self.module(self.a, self.s)

    def peakmem_conditioned_transition_block(self, batch_size, c, c_s, n, dtype):
        """Benchmark peak memory usage of ConditionedTransitionBlock."""
        return self.module(self.a, self.s)


class TimeConditionedTransitionBlockLarge:
    """Benchmark large-scale ConditionedTransitionBlock configurations."""

    params = ([1, 4], [256, 512], [384, 512], [torch.float32])
    param_names = ["batch_size", "c", "c_s", "dtype"]

    def setup(self, batch_size, c, c_s, dtype):
        device = torch.device("cpu")
        self.seq_len = 12  # Smaller sequence for large models
        self.n = 4  # Standard expansion factor

        # Create module and compile for optimal performance
        module = ConditionedTransitionBlock(c=c, c_s=c_s, n=self.n).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, self.seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, self.seq_len, c_s, dtype=dtype, device=device)

    def time_conditioned_transition_large(self, batch_size, c, c_s, dtype):
        """Benchmark forward pass of large ConditionedTransitionBlock."""
        return self.module(self.a, self.s)

    def peakmem_conditioned_transition_large(self, batch_size, c, c_s, dtype):
        """Benchmark peak memory usage of large ConditionedTransitionBlock."""
        return self.module(self.a, self.s)


class TimeConditionedTransitionBlockGradients:
    """Benchmark ConditionedTransitionBlock gradient computation."""

    params = ([1, 2], [64, 128], [64, 128], [torch.float32])
    param_names = ["batch_size", "c", "c_s", "dtype"]

    def setup(self, batch_size, c, c_s, dtype):
        device = torch.device("cpu")
        self.seq_len = 10
        self.n = 4

        # Create module (don't compile for gradient benchmarks)
        self.module = (
            ConditionedTransitionBlock(c=c, c_s=c_s, n=self.n).to(device).to(dtype)
        )

        # Generate test data
        self.a = torch.randn(
            batch_size, self.seq_len, c, dtype=dtype, device=device, requires_grad=True
        )
        self.s = torch.randn(
            batch_size,
            self.seq_len,
            c_s,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_conditioned_transition_backward(self, batch_size, c, c_s, dtype):
        """Benchmark backward pass of ConditionedTransitionBlock."""
        # Zero gradients
        if self.a.grad is not None:
            self.a.grad.zero_()
        if self.s.grad is not None:
            self.s.grad.zero_()

        # Forward pass
        a_out = self.module(self.a, self.s)
        loss = a_out.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_conditioned_transition_backward(self, batch_size, c, c_s, dtype):
        """Benchmark peak memory usage of ConditionedTransitionBlock backward pass."""
        # Zero gradients
        if self.a.grad is not None:
            self.a.grad.zero_()
        if self.s.grad is not None:
            self.s.grad.zero_()

        # Forward pass
        a_out = self.module(self.a, self.s)
        loss = a_out.sum()

        # Backward pass
        loss.backward()

        return loss


class TimeAdaptiveLayerNormScaling:
    """Benchmark AdaptiveLayerNorm scaling with sequence length."""

    params = ([5, 10, 20], [64], [128], [torch.float32])
    param_names = ["seq_len", "c", "c_s", "dtype"]

    def setup(self, seq_len, c, c_s, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module and compile for optimal performance
        module = AdaptiveLayerNorm(c=c, c_s=c_s).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)

    def time_adaptive_layer_norm_scaling(self, seq_len, c, c_s, dtype):
        """Benchmark AdaptiveLayerNorm scaling with sequence length."""
        return self.module(self.a, self.s)

    def peakmem_adaptive_layer_norm_scaling(self, seq_len, c, c_s, dtype):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.a, self.s)


class TimeConditionedTransitionBlockScaling:
    """Benchmark ConditionedTransitionBlock scaling with sequence length."""

    params = ([5, 10, 20], [64], [128], [torch.float32])
    param_names = ["seq_len", "c", "c_s", "dtype"]

    def setup(self, seq_len, c, c_s, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n = 4

        # Create module and compile for optimal performance
        module = ConditionedTransitionBlock(c=c, c_s=c_s, n=n).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)

    def time_conditioned_transition_scaling(self, seq_len, c, c_s, dtype):
        """Benchmark ConditionedTransitionBlock scaling with sequence length."""
        return self.module(self.a, self.s)

    def peakmem_conditioned_transition_scaling(self, seq_len, c, c_s, dtype):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.a, self.s)


class TimeAdaptiveLayerNormComparison:
    """Benchmark AdaptiveLayerNorm vs standard LayerNorm."""

    params = ([64, 128, 256], [torch.float32])
    param_names = ["c", "dtype"]

    def setup(self, c, dtype):
        device = torch.device("cpu")
        batch_size = 8
        seq_len = 20
        c_s = c  # Same size for fair comparison

        # AdaptiveLayerNorm
        self.adaptive_ln = torch.compile(
            AdaptiveLayerNorm(c=c, c_s=c_s).to(device).to(dtype), fullgraph=True
        )

        # Standard LayerNorm for comparison
        self.standard_ln = torch.compile(
            torch.nn.LayerNorm(c).to(device).to(dtype), fullgraph=True
        )

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)

    def time_adaptive_layer_norm_vs_standard(self, c, dtype):
        """Benchmark AdaptiveLayerNorm."""
        return self.adaptive_ln(self.a, self.s)

    def time_standard_layer_norm(self, c, dtype):
        """Benchmark standard LayerNorm for comparison."""
        return self.standard_ln(self.a)

    def peakmem_adaptive_layer_norm_vs_standard(self, c, dtype):
        """Benchmark AdaptiveLayerNorm memory."""
        return self.adaptive_ln(self.a, self.s)

    def peakmem_standard_layer_norm(self, c, dtype):
        """Benchmark standard LayerNorm memory."""
        return self.standard_ln(self.a)

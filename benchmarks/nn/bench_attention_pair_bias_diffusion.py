import os

import torch

from beignet.nn import AttentionPairBias

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeAttentionPairBiasDiffusion:
    """Benchmark AttentionPairBias (Algorithm 24) with diffusion features."""

    params = (
        [1, 2, 4],
        [8, 16, 32],
        [64, 128, 256],
        [64, 128, 256],
        [32, 64, 128],
        [4, 8, 16],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "seq_len", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module with conditioning and compile for optimal performance
        module = (
            AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Create module without conditioning for comparison
        module_no_cond = (
            AttentionPairBias(c_a=c_a, c_s=None, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module_no_cond = torch.compile(module_no_cond, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        self.z = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        self.beta = torch.randn(
            batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device
        )

    def time_attention_pair_bias_full(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark AttentionPairBias with all inputs (a, s, z, beta)."""
        return self.module(self.a, self.s, self.z, self.beta)

    def time_attention_pair_bias_no_conditioning(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark AttentionPairBias without conditioning (a, z, beta only)."""
        return self.module_no_cond(self.a, None, self.z, self.beta)

    def time_attention_pair_bias_minimal(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark AttentionPairBias with minimal inputs (a only)."""
        return self.module_no_cond(self.a)

    def peakmem_attention_pair_bias_full(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory of AttentionPairBias with all inputs."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_attention_pair_bias_no_conditioning(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory of AttentionPairBias without conditioning."""
        return self.module_no_cond(self.a, None, self.z, self.beta)


class TimeAttentionPairBiasLarge:
    """Benchmark large-scale AttentionPairBias configurations."""

    params = (
        [1, 2],
        [64, 128],
        [512, 1024],
        [512, 1024],
        [256, 512],
        [16],
        [torch.float32],
    )
    param_names = ["batch_size", "seq_len", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        self.z = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        self.beta = torch.randn(
            batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device
        )

    def time_attention_pair_bias_large(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark forward pass of large AttentionPairBias."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_attention_pair_bias_large(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory usage of large AttentionPairBias."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeAttentionPairBiasScaling:
    """Benchmark AttentionPairBias scaling with sequence length."""

    params = ([8, 16, 32, 64], [128], [128], [64], [8], [torch.float32])
    param_names = ["seq_len", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, seq_len, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module and compile for optimal performance
        module = (
            AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        self.z = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        self.beta = torch.randn(
            batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device
        )

    def time_attention_pair_bias_scaling(self, seq_len, c_a, c_s, c_z, n_head, dtype):
        """Benchmark AttentionPairBias scaling with sequence length."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_attention_pair_bias_scaling(
        self, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeAttentionPairBiasGradients:
    """Benchmark AttentionPairBias gradient computation."""

    params = ([1, 2], [16, 32], [64, 128], [64, 128], [32, 64], [8], [torch.float32])
    param_names = ["batch_size", "seq_len", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module (don't compile for gradient benchmarks)
        self.module = (
            AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.a = torch.randn(
            batch_size, seq_len, c_a, dtype=dtype, device=device, requires_grad=True
        )
        self.s = torch.randn(
            batch_size, seq_len, c_s, dtype=dtype, device=device, requires_grad=True
        )
        self.z = torch.randn(
            batch_size,
            seq_len,
            seq_len,
            c_z,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.beta = torch.randn(
            batch_size,
            seq_len,
            seq_len,
            n_head,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_attention_pair_bias_backward(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark backward pass of AttentionPairBias."""
        # Zero gradients
        for tensor in [self.a, self.s, self.z, self.beta]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        a_out = self.module(self.a, self.s, self.z, self.beta)
        loss = a_out.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_attention_pair_bias_backward(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory usage of AttentionPairBias backward pass."""
        # Zero gradients
        for tensor in [self.a, self.s, self.z, self.beta]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        a_out = self.module(self.a, self.s, self.z, self.beta)
        loss = a_out.sum()

        # Backward pass
        loss.backward()

        return loss


class TimeAttentionPairBiasHeadComparison:
    """Benchmark AttentionPairBias with different head counts."""

    params = ([4, 8, 16, 32], [128], [128], [64], [torch.float32])
    param_names = ["n_head", "c_a", "c_s", "c_z", "dtype"]

    def setup(self, n_head, c_a, c_s, c_z, dtype):
        device = torch.device("cpu")
        batch_size, seq_len = 2, 16

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
        self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        self.z = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        self.beta = torch.randn(
            batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device
        )

    def time_attention_pair_bias_heads(self, n_head, c_a, c_s, c_z, dtype):
        """Benchmark AttentionPairBias with different head counts."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_attention_pair_bias_heads(self, n_head, c_a, c_s, c_z, dtype):
        """Benchmark peak memory with different head counts."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeAttentionPairBiasConditioningComparison:
    """Compare AttentionPairBias with and without conditioning."""

    params = (
        ["with_conditioning", "without_conditioning"],
        [64],
        [64],
        [32],
        [8],
        [torch.float32],
    )
    param_names = ["conditioning_type", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, conditioning_type, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")
        batch_size, seq_len = 4, 20

        if conditioning_type == "with_conditioning":
            # Create module with conditioning
            module = (
                AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
                .to(device)
                .to(dtype)
            )
            self.module = torch.compile(module, fullgraph=True)
            # Generate conditioning signal
            self.s = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        else:
            # Create module without conditioning
            module = (
                AttentionPairBias(c_a=c_a, c_s=None, c_z=c_z, n_head=n_head)
                .to(device)
                .to(dtype)
            )
            self.module = torch.compile(module, fullgraph=True)
            self.s = None

        # Generate test data
        self.a = torch.randn(batch_size, seq_len, c_a, dtype=dtype, device=device)
        self.z = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )
        self.beta = torch.randn(
            batch_size, seq_len, seq_len, n_head, dtype=dtype, device=device
        )

    def time_attention_pair_bias_conditioning(
        self, conditioning_type, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark AttentionPairBias with/without conditioning."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_attention_pair_bias_conditioning(
        self, conditioning_type, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory with/without conditioning."""
        return self.module(self.a, self.s, self.z, self.beta)

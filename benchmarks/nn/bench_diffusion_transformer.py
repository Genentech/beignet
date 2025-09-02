import os

import torch

from beignet.nn import DiffusionTransformer

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeDiffusionTransformer:
    """Benchmark DiffusionTransformer (Algorithm 23)."""

    params = (
        [1, 2, 4],
        [8, 16, 32],
        [64, 128, 256],
        [64, 128, 256],
        [32, 64, 128],
        [4, 8, 16],
        [1, 2, 4],
        [torch.float32, torch.float64],
    )
    param_names = [
        "batch_size",
        "seq_len",
        "c_a",
        "c_s",
        "c_z",
        "n_head",
        "n_block",
        "dtype",
    ]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block
            )
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

    def time_diffusion_transformer(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark forward pass of DiffusionTransformer."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_diffusion_transformer(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark peak memory usage of DiffusionTransformer."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeDiffusionTransformerLarge:
    """Benchmark large-scale DiffusionTransformer configurations."""

    params = (
        [1, 2],
        [64, 128],
        [512, 1024],
        [512, 1024],
        [256, 512],
        [16],
        [2, 4],
        [torch.float32],
    )
    param_names = [
        "batch_size",
        "seq_len",
        "c_a",
        "c_s",
        "c_z",
        "n_head",
        "n_block",
        "dtype",
    ]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block
            )
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

    def time_diffusion_transformer_large(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark forward pass of large DiffusionTransformer."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_diffusion_transformer_large(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark peak memory usage of large DiffusionTransformer."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeDiffusionTransformerScaling:
    """Benchmark DiffusionTransformer scaling with sequence length."""

    params = ([8, 16, 32, 64], [128], [128], [64], [8], [2], [torch.float32])
    param_names = ["seq_len", "c_a", "c_s", "c_z", "n_head", "n_block", "dtype"]

    def setup(self, seq_len, c_a, c_s, c_z, n_head, n_block, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module and compile for optimal performance
        module = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block
            )
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

    def time_diffusion_transformer_scaling(
        self, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark DiffusionTransformer scaling with sequence length."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_diffusion_transformer_scaling(
        self, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.a, self.s, self.z, self.beta)


class TimeDiffusionTransformerGradients:
    """Benchmark DiffusionTransformer gradient computation."""

    params = (
        [1, 2],
        [16, 32],
        [64, 128],
        [64, 128],
        [32, 64],
        [8],
        [1, 2],
        [torch.float32],
    )
    param_names = [
        "batch_size",
        "seq_len",
        "c_a",
        "c_s",
        "c_z",
        "n_head",
        "n_block",
        "dtype",
    ]

    def setup(self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype):
        device = torch.device("cpu")

        # Ensure c_a is divisible by n_head
        if c_a % n_head != 0:
            c_a = (c_a // n_head + 1) * n_head

        # Create module (don't compile for gradient benchmarks)
        self.module = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block
            )
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

    def time_diffusion_transformer_backward(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark backward pass of DiffusionTransformer."""
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

    def peakmem_diffusion_transformer_backward(
        self, batch_size, seq_len, c_a, c_s, c_z, n_head, n_block, dtype
    ):
        """Benchmark peak memory usage of DiffusionTransformer backward pass."""
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


class TimeDiffusionTransformerBlockComparison:
    """Benchmark DiffusionTransformer with different block counts."""

    params = ([1, 2, 4, 8], [64], [64], [32], [8], [torch.float32])
    param_names = ["n_block", "c_a", "c_s", "c_z", "n_head", "dtype"]

    def setup(self, n_block, c_a, c_s, c_z, n_head, dtype):
        device = torch.device("cpu")
        batch_size, seq_len = 2, 16

        # Create module and compile for optimal performance
        module = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head, n_block=n_block
            )
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

    def time_diffusion_transformer_blocks(self, n_block, c_a, c_s, c_z, n_head, dtype):
        """Benchmark DiffusionTransformer with different block counts."""
        return self.module(self.a, self.s, self.z, self.beta)

    def peakmem_diffusion_transformer_blocks(
        self, n_block, c_a, c_s, c_z, n_head, dtype
    ):
        """Benchmark peak memory with different block counts."""
        return self.module(self.a, self.s, self.z, self.beta)

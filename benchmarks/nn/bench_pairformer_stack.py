import os

import torch

from beignet.nn import AttentionPairBias, PairformerStack, PairformerStackBlock

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeAttentionPairBias:
    """Benchmark AttentionPairBias module."""

    params = ([1, 4, 16], [64, 128, 256], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 20

        # Create module and compile for optimal performance
        module = AttentionPairBias(c_s=c_s, c_z=c_z, n_head=16).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, self.seq_len, self.seq_len, c_z, dtype=dtype, device=device
        )

    def time_attention_pair_bias(self, batch_size, c_s, c_z, dtype):
        """Benchmark forward pass of AttentionPairBias."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_attention_pair_bias(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of AttentionPairBias."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerStackBlock:
    """Benchmark PairformerStackBlock module (Algorithm 17 implementation)."""

    params = ([1, 4, 16], [64, 128], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 15

        # Create module with Algorithm 17 specifications
        module = (
            PairformerStackBlock(
                c_s=c_s, c_z=c_z, n_head_single=16, n_head_pair=4, dropout_rate=0.25
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, self.seq_len, self.seq_len, c_z, dtype=dtype, device=device
        )

    def time_pairformer_stack_block(self, batch_size, c_s, c_z, dtype):
        """Benchmark forward pass of PairformerStackBlock."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_stack_block(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of PairformerStackBlock."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerStack:
    """Benchmark PairformerStack module (Algorithm 17 full implementation)."""

    params = ([1, 4], [2, 4, 8], [64, 128], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_block", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, n_block, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 12  # Smaller sequence length for larger models

        # Create module with Algorithm 17 specifications
        module = (
            PairformerStack(
                n_block=n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16,  # Algorithm 17 specification
                n_head_pair=4,
                dropout_rate=0.25,  # Algorithm 17 specification
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, self.seq_len, self.seq_len, c_z, dtype=dtype, device=device
        )

    def time_pairformer_stack(self, batch_size, n_block, c_s, c_z, dtype):
        """Benchmark forward pass of PairformerStack."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_stack(self, batch_size, n_block, c_s, c_z, dtype):
        """Benchmark peak memory usage of PairformerStack."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerStackLarge:
    """Benchmark large-scale PairformerStack configurations (Algorithm 17)."""

    params = ([1, 2], [256, 384], [64, 128], [torch.float32])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 8  # Small sequence for large models
        self.n_block = 4  # Fewer blocks for large models

        # Create module with Algorithm 17 specifications
        module = (
            PairformerStack(
                n_block=self.n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16,  # Algorithm 17 specification
                n_head_pair=8 if c_z >= 64 else 4,
                dropout_rate=0.25,  # Algorithm 17 specification
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, self.seq_len, self.seq_len, c_z, dtype=dtype, device=device
        )

    def time_pairformer_stack_large(self, batch_size, c_s, c_z, dtype):
        """Benchmark forward pass of large PairformerStack."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_stack_large(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of large PairformerStack."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerStackGradients:
    """Benchmark PairformerStack gradient computation (Algorithm 17)."""

    params = ([1, 2], [64, 128], [32, 64], [torch.float32])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 10
        self.n_block = 2

        # Create module with Algorithm 17 specifications (don't compile for gradient benchmarks)
        self.module = (
            PairformerStack(
                n_block=self.n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16,
                n_head_pair=4,
                dropout_rate=0.25,
            )
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.s_i = torch.randn(
            batch_size,
            self.seq_len,
            c_s,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.z_ij = torch.randn(
            batch_size,
            self.seq_len,
            self.seq_len,
            c_z,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_pairformer_stack_backward(self, batch_size, c_s, c_z, dtype):
        """Benchmark backward pass of PairformerStack."""
        # Zero gradients
        if self.s_i.grad is not None:
            self.s_i.grad.zero_()
        if self.z_ij.grad is not None:
            self.z_ij.grad.zero_()

        # Forward pass
        s_out, z_out = self.module(self.s_i, self.z_ij)
        loss = s_out.sum() + z_out.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_pairformer_stack_backward(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of PairformerStack backward pass."""
        # Zero gradients
        if self.s_i.grad is not None:
            self.s_i.grad.zero_()
        if self.z_ij.grad is not None:
            self.z_ij.grad.zero_()

        # Forward pass
        s_out, z_out = self.module(self.s_i, self.z_ij)
        loss = s_out.sum() + z_out.sum()

        # Backward pass
        loss.backward()

        return loss


class TimePairformerStackScaling:
    """Benchmark PairformerStack scaling with sequence length (Algorithm 17)."""

    params = ([5, 10, 20], [64], [32], [torch.float32])
    param_names = ["seq_len", "c_s", "c_z", "dtype"]

    def setup(self, seq_len, c_s, c_z, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n_block = 2

        # Create module with Algorithm 17 specifications
        module = (
            PairformerStack(
                n_block=n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16,
                n_head_pair=4,
                dropout_rate=0.25,
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(batch_size, seq_len, c_s, dtype=dtype, device=device)
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )

    def time_pairformer_stack_scaling(self, seq_len, c_s, c_z, dtype):
        """Benchmark PairformerStack scaling with sequence length."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_stack_scaling(self, seq_len, c_s, c_z, dtype):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerStackFullScale:
    """Benchmark PairformerStack with Algorithm 17 default parameters (N_block=48)."""

    params = ([1], [384], [128], [torch.float32])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 6  # Very small sequence for full scale model
        self.n_block = 48  # Algorithm 17 default

        # Create module with exact Algorithm 17 specifications
        module = (
            PairformerStack(
                n_block=self.n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16,  # Algorithm 17 specification
                n_head_pair=4,
                dropout_rate=0.25,  # Algorithm 17 specification
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, self.seq_len, self.seq_len, c_z, dtype=dtype, device=device
        )

    def time_pairformer_stack_full_scale(self, batch_size, c_s, c_z, dtype):
        """Benchmark full-scale PairformerStack (48 blocks)."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_stack_full_scale(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of full-scale PairformerStack."""
        return self.module(self.s_i, self.z_ij)

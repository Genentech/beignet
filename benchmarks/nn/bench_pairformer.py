import os

import torch

from beignet.nn import Pairformer, PairformerBlock, SingleRowAttention

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeSingleRowAttention:
    """Benchmark SingleRowAttention module."""

    params = ([1, 4, 16], [32, 64, 128], [torch.float32, torch.float64])
    param_names = ["batch_size", "c_s", "dtype"]

    def setup(self, batch_size, c_s, dtype):
        device = torch.device("cpu")
        self.seq_len = 20
        self.n_head = 8

        # Create module and compile for optimal performance
        module = SingleRowAttention(c_s=c_s, n_head=self.n_head).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.s_i = torch.randn(
            batch_size, self.seq_len, c_s, dtype=dtype, device=device
        )

    def time_single_row_attention(self, batch_size, c_s, dtype):
        """Benchmark forward pass of SingleRowAttention."""
        return self.module(self.s_i)

    def peakmem_single_row_attention(self, batch_size, c_s, dtype):
        """Benchmark peak memory usage of SingleRowAttention."""
        return self.module(self.s_i)


class TimePairformerBlock:
    """Benchmark PairformerBlock module."""

    params = ([1, 4, 16], [32, 64], [16, 32], [torch.float32, torch.float64])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 15

        # Create module and compile for optimal performance
        module = (
            PairformerBlock(c_s=c_s, c_z=c_z, n_head_single=8, n_head_pair=4)
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

    def time_pairformer_block(self, batch_size, c_s, c_z, dtype):
        """Benchmark forward pass of PairformerBlock."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_block(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of PairformerBlock."""
        return self.module(self.s_i, self.z_ij)


class TimePairformer:
    """Benchmark Pairformer module."""

    params = ([1, 4], [2, 4, 8], [64, 128], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_block", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, n_block, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 12  # Smaller sequence length for larger models

        # Create module and compile for optimal performance
        module = (
            Pairformer(
                n_block=n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=8,
                n_head_pair=4,
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

    def time_pairformer(self, batch_size, n_block, c_s, c_z, dtype):
        """Benchmark forward pass of Pairformer."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer(self, batch_size, n_block, c_s, c_z, dtype):
        """Benchmark peak memory usage of Pairformer."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerLarge:
    """Benchmark large-scale Pairformer configurations."""

    params = ([1, 2], [384, 512], [128, 192], [torch.float32])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 8  # Small sequence for large models
        self.n_block = 4  # Fewer blocks for large models

        # Create module and compile for optimal performance
        module = (
            Pairformer(
                n_block=self.n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=16 if c_s >= 384 else 8,
                n_head_pair=8 if c_z >= 128 else 4,
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

    def time_pairformer_large(self, batch_size, c_s, c_z, dtype):
        """Benchmark forward pass of large Pairformer."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_large(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of large Pairformer."""
        return self.module(self.s_i, self.z_ij)


class TimePairformerGradients:
    """Benchmark Pairformer gradient computation."""

    params = ([1, 2], [32, 64], [16, 32], [torch.float32])
    param_names = ["batch_size", "c_s", "c_z", "dtype"]

    def setup(self, batch_size, c_s, c_z, dtype):
        device = torch.device("cpu")
        self.seq_len = 10
        self.n_block = 2

        # Create module (don't compile for gradient benchmarks)
        self.module = (
            Pairformer(
                n_block=self.n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=8,
                n_head_pair=4,
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

    def time_pairformer_backward(self, batch_size, c_s, c_z, dtype):
        """Benchmark backward pass of Pairformer."""
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

    def peakmem_pairformer_backward(self, batch_size, c_s, c_z, dtype):
        """Benchmark peak memory usage of Pairformer backward pass."""
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


class TimePairformerScaling:
    """Benchmark Pairformer scaling with sequence length."""

    params = ([5, 10, 20], [32], [16], [torch.float32])
    param_names = ["seq_len", "c_s", "c_z", "dtype"]

    def setup(self, seq_len, c_s, c_z, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n_block = 2

        # Create module and compile for optimal performance
        module = (
            Pairformer(
                n_block=n_block,
                c_s=c_s,
                c_z=c_z,
                n_head_single=8,
                n_head_pair=4,
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

    def time_pairformer_scaling(self, seq_len, c_s, c_z, dtype):
        """Benchmark Pairformer scaling with sequence length."""
        return self.module(self.s_i, self.z_ij)

    def peakmem_pairformer_scaling(self, seq_len, c_s, c_z, dtype):
        """Benchmark peak memory scaling with sequence length."""
        return self.module(self.s_i, self.z_ij)

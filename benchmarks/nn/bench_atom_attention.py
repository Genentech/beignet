import os

import torch

from beignet.nn import (
    AtomAttentionEncoder,
    RelativePositionEncoding,
    _AtomAttentionDecoder,
)

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeAtomAttentionEncoder:
    """Benchmark AtomAttentionEncoder."""

    params = (
        [1, 2, 4],
        [8, 16, 32],
        [32, 64, 128],
        [x for x in [128, 256, 512] if x % 16 == 0],  # Ensure divisible by n_head
        [32, 64, 128],
        [8, 16, 32],
        [4, 8, 16],
        [torch.float32, torch.float64],
    )
    param_names = [
        "batch_size",
        "n_tokens",
        "n_atoms",
        "c_token",
        "c_atom",
        "c_atompair",
        "n_head",
        "dtype",
    ]

    def setup(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        device = torch.device("cpu")

        # Ensure c_token is divisible by n_head
        if c_token % n_head != 0:
            c_token = (c_token // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            AtomAttentionEncoder(
                c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, n_head=n_head
            )
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.r_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, c_atompair, dtype=dtype, device=device
        )

    def time_atom_attention_encoder(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        """Benchmark forward pass of AtomAttentionEncoder."""
        return self.module(self.f_star, self.r_noisy, self.s_trunk, self.z_atom)

    def peakmem_atom_attention_encoder(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        """Benchmark peak memory usage of AtomAttentionEncoder."""
        return self.module(self.f_star, self.r_noisy, self.s_trunk, self.z_atom)


class TimeAtomAttentionDecoder:
    """Benchmark AtomAttentionDecoder."""

    params = (
        [1, 2, 4],
        [8, 16, 32],
        [32, 64, 128],
        [x for x in [128, 256, 512] if x % 16 == 0],  # Ensure divisible by n_head
        [x for x in [32, 64, 128] if x % 16 == 0],  # Ensure divisible by n_head
        [8, 16, 32],
        [4, 8, 16],
        [torch.float32, torch.float64],
    )
    param_names = [
        "batch_size",
        "n_tokens",
        "n_atoms",
        "c_token",
        "c_atom",
        "c_atompair",
        "n_head",
        "dtype",
    ]

    def setup(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        device = torch.device("cpu")

        # Ensure dimensions are divisible by n_head
        if c_token % n_head != 0:
            c_token = (c_token // n_head + 1) * n_head
        if c_atom % n_head != 0:
            c_atom = (c_atom // n_head + 1) * n_head

        # Create module and compile for optimal performance
        module = (
            _AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.a = torch.randn(batch_size, n_tokens, c_token, dtype=dtype, device=device)
        self.q_skip = torch.randn(
            batch_size, n_atoms, c_token, dtype=dtype, device=device
        )
        self.c_skip = torch.randn(
            batch_size, n_atoms, c_atom, dtype=dtype, device=device
        )
        self.p_skip = torch.randn(
            batch_size, n_atoms, n_atoms, c_atompair, dtype=dtype, device=device
        )

    def time_atom_attention_decoder(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        """Benchmark forward pass of AtomAttentionDecoder."""
        return self.module(self.a, self.q_skip, self.c_skip, self.p_skip)

    def peakmem_atom_attention_decoder(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, c_atompair, n_head, dtype
    ):
        """Benchmark peak memory usage of AtomAttentionDecoder."""
        return self.module(self.a, self.q_skip, self.c_skip, self.p_skip)


class TimeRelativePositionEncoding:
    """Benchmark RelativePositionEncoding."""

    params = (
        [1, 2, 4],
        [16, 32, 64],
        [32, 64, 128],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "n_atoms", "c_out", "dtype"]

    def setup(self, batch_size, n_atoms, c_out, dtype):
        device = torch.device("cpu")

        # Create module and compile for optimal performance
        module = RelativePositionEncoding(c_out=c_out).to(device).to(dtype)
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.positions = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)

    def time_relative_position_encoding(self, batch_size, n_atoms, c_out, dtype):
        """Benchmark forward pass of RelativePositionEncoding."""
        return self.module(self.positions)

    def peakmem_relative_position_encoding(self, batch_size, n_atoms, c_out, dtype):
        """Benchmark peak memory usage of RelativePositionEncoding."""
        return self.module(self.positions)


class TimeAtomAttentionScaling:
    """Benchmark atom attention scaling with atom count."""

    params = ([16, 32, 64, 128], [64], [32], [8], [torch.float32])
    param_names = ["n_atoms", "c_token", "c_atom", "n_head", "dtype"]

    def setup(self, n_atoms, c_token, c_atom, n_head, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n_tokens = 8

        # Create modules
        self.encoder = (
            AtomAttentionEncoder(
                c_token=c_token, c_atom=c_atom, c_atompair=16, n_head=n_head
            )
            .to(device)
            .to(dtype)
        )

        self.decoder = (
            _AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, n_head=n_head)
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.r_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, 16, dtype=dtype, device=device
        )

        # Pre-compute encoder outputs for decoder
        with torch.no_grad():
            self.a, self.q_skip, self.c_skip, self.p_skip = self.encoder(
                self.f_star, self.r_noisy, self.s_trunk, self.z_atom
            )

    def time_atom_attention_encoder_scaling(
        self, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark AtomAttentionEncoder scaling with atom count."""
        return self.encoder(self.f_star, self.r_noisy, self.s_trunk, self.z_atom)

    def time_atom_attention_decoder_scaling(
        self, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark AtomAttentionDecoder scaling with atom count."""
        return self.decoder(self.a, self.q_skip, self.c_skip, self.p_skip)

    def peakmem_atom_attention_encoder_scaling(
        self, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark memory scaling for AtomAttentionEncoder."""
        return self.encoder(self.f_star, self.r_noisy, self.s_trunk, self.z_atom)

    def peakmem_atom_attention_decoder_scaling(
        self, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark memory scaling for AtomAttentionDecoder."""
        return self.decoder(self.a, self.q_skip, self.c_skip, self.p_skip)


class TimeAtomAttentionGradients:
    """Benchmark atom attention gradient computation."""

    params = ([1, 2], [8, 16], [16, 32], [64], [32], [8], [torch.float32])
    param_names = [
        "batch_size",
        "n_tokens",
        "n_atoms",
        "c_token",
        "c_atom",
        "n_head",
        "dtype",
    ]

    def setup(self, batch_size, n_tokens, n_atoms, c_token, c_atom, n_head, dtype):
        device = torch.device("cpu")

        # Create modules (don't compile for gradient benchmarks)
        self.encoder = (
            AtomAttentionEncoder(
                c_token=c_token, c_atom=c_atom, c_atompair=16, n_head=n_head
            )
            .to(device)
            .to(dtype)
        )

        self.decoder = (
            _AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, n_head=n_head)
            .to(device)
            .to(dtype)
        )

        # Generate test data with gradients
        self.f_star = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device, requires_grad=True
        )
        self.r_noisy = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device, requires_grad=True
        )
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device, requires_grad=True
        )
        self.z_atom = torch.randn(
            batch_size,
            n_atoms,
            n_atoms,
            16,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        # Decoder inputs
        self.a = torch.randn(
            batch_size,
            n_tokens,
            c_token,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.q_skip = torch.randn(
            batch_size, n_atoms, c_token, dtype=dtype, device=device, requires_grad=True
        )
        self.c_skip = torch.randn(
            batch_size, n_atoms, c_atom, dtype=dtype, device=device, requires_grad=True
        )
        self.p_skip = torch.randn(
            batch_size,
            n_atoms,
            n_atoms,
            16,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_atom_attention_encoder_backward(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark backward pass of AtomAttentionEncoder."""
        # Zero gradients
        for tensor in [self.f_star, self.r_noisy, self.s_trunk, self.z_atom]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        a, q_skip, c_skip, p_skip = self.encoder(
            self.f_star, self.r_noisy, self.s_trunk, self.z_atom
        )
        loss = a.sum() + q_skip.sum() + c_skip.sum() + p_skip.sum()

        # Backward pass
        loss.backward()

        return loss

    def time_atom_attention_decoder_backward(
        self, batch_size, n_tokens, n_atoms, c_token, c_atom, n_head, dtype
    ):
        """Benchmark backward pass of AtomAttentionDecoder."""
        # Zero gradients
        for tensor in [self.a, self.q_skip, self.c_skip, self.p_skip]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        r_update = self.decoder(self.a, self.q_skip, self.c_skip, self.p_skip)
        loss = r_update.sum()

        # Backward pass
        loss.backward()

        return loss

import os

import torch

from beignet.nn import _Diffusion

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeAlphaFold3Diffusion:
    """Benchmark AlphaFold3Diffusion (Algorithm 20)."""

    params = (
        [1, 2],
        [8, 16],
        [32, 64],
        [torch.float32],  # Only float32 due to complexity
    )
    param_names = ["batch_size", "n_tokens", "n_atoms", "dtype"]

    def setup(self, batch_size, n_tokens, n_atoms, dtype):
        device = torch.device("cpu")

        # Create module with smaller parameters for benchmarking
        module = (
            _Diffusion(
                c_token=256,  # Smaller than default for faster benchmarks
                c_atom=64,
                c_atompair=16,
                n_head=8,
                n_block=4,  # Much smaller than default 24 for benchmarking
            )
            .to(device)
            .to(dtype)
        )

        # Don't compile for benchmarks due to complexity
        self.module = module

        # Generate test data
        self.x_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, 100, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_trunk = torch.randn(
            batch_size, n_tokens, n_tokens, 128, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, 16, dtype=dtype, device=device
        )

    def time_diffusion_module(self, batch_size, n_tokens, n_atoms, dtype):
        """Benchmark forward pass of AlphaFold3Diffusion."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )

    def peakmem_diffusion_module(self, batch_size, n_tokens, n_atoms, dtype):
        """Benchmark peak memory usage of AlphaFold3Diffusion."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )


class TimeAlphaFold3DiffusionComponents:
    """Benchmark individual components of AlphaFold3Diffusion."""

    params = ([8, 16], [32, 64], [torch.float32])
    param_names = ["n_tokens", "n_atoms", "dtype"]

    def setup(self, n_tokens, n_atoms, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module with smaller parameters
        self.module = (
            _Diffusion(c_token=256, c_atom=64, c_atompair=16, n_head=8, n_block=2)
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.x_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, 100, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_trunk = torch.randn(
            batch_size, n_tokens, n_tokens, 128, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, 16, dtype=dtype, device=device
        )

    def time_diffusion_conditioning_component(self, n_tokens, n_atoms, dtype):
        """Benchmark DiffusionConditioning component."""
        return self.module.diffusion_conditioning(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )

    def time_atom_attention_encoder_component(self, n_tokens, n_atoms, dtype):
        """Benchmark AtomAttentionEncoder component."""
        return self.module.atom_attention_encoder(
            self.f_star, self.x_noisy, self.s_trunk, self.z_atom
        )

    def time_atom_attention_decoder_component(self, n_tokens, n_atoms, dtype):
        """Benchmark AtomAttentionDecoder component."""
        # Create dummy inputs for decoder
        a = torch.randn(
            1, n_tokens, 256, device=self.x_noisy.device, dtype=self.x_noisy.dtype
        )
        q_skip = torch.randn(
            1, n_atoms, 256, device=self.x_noisy.device, dtype=self.x_noisy.dtype
        )
        c_skip = torch.randn(
            1, n_atoms, 64, device=self.x_noisy.device, dtype=self.x_noisy.dtype
        )
        p_skip = torch.randn(
            1,
            n_atoms,
            n_atoms,
            16,
            device=self.x_noisy.device,
            dtype=self.x_noisy.dtype,
        )

        return self.module.atom_attention_decoder(a, q_skip, c_skip, p_skip)


class TimeAlphaFold3DiffusionScaling:
    """Benchmark AlphaFold3Diffusion scaling with atom count."""

    params = ([16, 32, 64], [torch.float32])
    param_names = ["n_atoms", "dtype"]

    def setup(self, n_atoms, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n_tokens = 8

        # Create module with minimal parameters for scaling tests
        self.module = (
            _Diffusion(c_token=128, c_atom=32, c_atompair=8, n_head=4, n_block=1)
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.x_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(batch_size, n_atoms, 64, dtype=dtype, device=device)
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_trunk = torch.randn(
            batch_size, n_tokens, n_tokens, 128, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, 8, dtype=dtype, device=device
        )

    def time_diffusion_module_scaling(self, n_atoms, dtype):
        """Benchmark AlphaFold3Diffusion scaling with atom count."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )

    def peakmem_diffusion_module_scaling(self, n_atoms, dtype):
        """Benchmark peak memory scaling with atom count."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )


class TimeAlphaFold3DiffusionGradients:
    """Benchmark AlphaFold3Diffusion gradient computation."""

    params = ([4, 8], [16, 32], [torch.float32])
    param_names = ["n_tokens", "n_atoms", "dtype"]

    def setup(self, n_tokens, n_atoms, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module with minimal parameters for gradient tests
        self.module = (
            _Diffusion(c_token=64, c_atom=32, c_atompair=8, n_head=4, n_block=1)
            .to(device)
            .to(dtype)
        )

        # Generate test data with gradients
        self.x_noisy = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device, requires_grad=True
        )
        self.t = (
            torch.randn(
                batch_size, 1, dtype=dtype, device=device, requires_grad=True
            ).abs()
            + 0.1
        )
        self.f_star = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device, requires_grad=True
        )
        self.s_inputs = torch.randn(
            batch_size, n_atoms, 64, dtype=dtype, device=device, requires_grad=True
        )
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device, requires_grad=True
        )
        self.z_trunk = torch.randn(
            batch_size,
            n_tokens,
            n_tokens,
            128,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.z_atom = torch.randn(
            batch_size,
            n_atoms,
            n_atoms,
            8,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_diffusion_module_backward(self, n_tokens, n_atoms, dtype):
        """Benchmark backward pass of AlphaFold3Diffusion."""
        # Zero gradients
        for tensor in [
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        ]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        x_out = self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )
        loss = x_out.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_diffusion_module_backward(self, n_tokens, n_atoms, dtype):
        """Benchmark peak memory usage of AlphaFold3Diffusion backward pass."""
        # Zero gradients
        for tensor in [
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        ]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        x_out = self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )
        loss = x_out.sum()

        # Backward pass
        loss.backward()

        return loss


class TimeAlphaFold3DiffusionTimesteps:
    """Benchmark AlphaFold3Diffusion with different timestep values."""

    params = (["small", "medium", "large"], [16], [torch.float32])
    param_names = ["timestep_scale", "n_atoms", "dtype"]

    def setup(self, timestep_scale, n_atoms, dtype):
        device = torch.device("cpu")
        batch_size = 1
        n_tokens = 4

        # Create module with minimal parameters
        self.module = (
            _Diffusion(c_token=64, c_atom=32, c_atompair=8, n_head=4, n_block=1)
            .to(device)
            .to(dtype)
        )

        # Generate test data with different timestep scales
        if timestep_scale == "small":
            self.t = torch.full((batch_size, 1), 0.01, dtype=dtype, device=device)
        elif timestep_scale == "medium":
            self.t = torch.full((batch_size, 1), 1.0, dtype=dtype, device=device)
        elif timestep_scale == "large":
            self.t = torch.full((batch_size, 1), 100.0, dtype=dtype, device=device)

        self.x_noisy = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(batch_size, n_atoms, 32, dtype=dtype, device=device)
        self.s_trunk = torch.randn(
            batch_size, n_tokens, 384, dtype=dtype, device=device
        )
        self.z_trunk = torch.randn(
            batch_size, n_tokens, n_tokens, 128, dtype=dtype, device=device
        )
        self.z_atom = torch.randn(
            batch_size, n_atoms, n_atoms, 8, dtype=dtype, device=device
        )

    def time_diffusion_module_timesteps(self, timestep_scale, n_atoms, dtype):
        """Benchmark AlphaFold3Diffusion with different timestep values."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )

    def peakmem_diffusion_module_timesteps(self, timestep_scale, n_atoms, dtype):
        """Benchmark peak memory with different timestep values."""
        return self.module(
            self.x_noisy,
            self.t,
            self.f_star,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
            self.z_atom,
        )

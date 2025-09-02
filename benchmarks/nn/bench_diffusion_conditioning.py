import os

import torch

from beignet.nn import DiffusionConditioning

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeDiffusionConditioning:
    """Benchmark DiffusionConditioning (Algorithm 21)."""

    params = (
        [1, 2, 4],
        [16, 32, 64],
        [64, 128, 256],
        [64, 128, 256],
        [32, 64, 128],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "n_atoms", "c_z", "c_s", "c_s_inputs", "dtype"]

    def setup(self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype):
        device = torch.device("cpu")

        # Create module and compile for optimal performance
        module = (
            DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, c_s_inputs, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(batch_size, n_atoms, c_s, dtype=dtype, device=device)
        self.z_trunk = torch.randn(
            batch_size, n_atoms, n_atoms, c_z, dtype=dtype, device=device
        )

    def time_diffusion_conditioning(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark forward pass of DiffusionConditioning."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )

    def peakmem_diffusion_conditioning(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark peak memory usage of DiffusionConditioning."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )


class TimeDiffusionConditioningLarge:
    """Benchmark large-scale DiffusionConditioning configurations."""

    params = ([1, 2], [128, 256], [128, 256], [384, 512], [128, 256], [torch.float32])
    param_names = ["batch_size", "n_atoms", "c_z", "c_s", "c_s_inputs", "dtype"]

    def setup(self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype):
        device = torch.device("cpu")

        # Create module and compile for optimal performance
        module = (
            DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, c_s_inputs, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(batch_size, n_atoms, c_s, dtype=dtype, device=device)
        self.z_trunk = torch.randn(
            batch_size, n_atoms, n_atoms, c_z, dtype=dtype, device=device
        )

    def time_diffusion_conditioning_large(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark forward pass of large DiffusionConditioning."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )

    def peakmem_diffusion_conditioning_large(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark peak memory usage of large DiffusionConditioning."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )


class TimeDiffusionConditioningScaling:
    """Benchmark DiffusionConditioning scaling with atom count."""

    params = ([16, 32, 64, 128], [128], [384], [128], [torch.float32])
    param_names = ["n_atoms", "c_z", "c_s", "c_s_inputs", "dtype"]

    def setup(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        device = torch.device("cpu")
        batch_size = 1

        # Create module and compile for optimal performance
        module = (
            DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
            .to(device)
            .to(dtype)
        )
        self.module = torch.compile(module, fullgraph=True)

        # Generate test data
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, c_s_inputs, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(batch_size, n_atoms, c_s, dtype=dtype, device=device)
        self.z_trunk = torch.randn(
            batch_size, n_atoms, n_atoms, c_z, dtype=dtype, device=device
        )

    def time_diffusion_conditioning_scaling(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        """Benchmark DiffusionConditioning scaling with atom count."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )

    def peakmem_diffusion_conditioning_scaling(
        self, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark peak memory scaling with atom count."""
        return self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )


class TimeDiffusionConditioningGradients:
    """Benchmark DiffusionConditioning gradient computation."""

    params = ([1, 2], [32, 64], [64, 128], [128, 256], [64, 128], [torch.float32])
    param_names = ["batch_size", "n_atoms", "c_z", "c_s", "c_s_inputs", "dtype"]

    def setup(self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype):
        device = torch.device("cpu")

        # Create module (don't compile for gradient benchmarks)
        self.module = (
            DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
            .to(device)
            .to(dtype)
        )

        # Generate test data
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
            batch_size,
            n_atoms,
            c_s_inputs,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.s_trunk = torch.randn(
            batch_size, n_atoms, c_s, dtype=dtype, device=device, requires_grad=True
        )
        self.z_trunk = torch.randn(
            batch_size,
            n_atoms,
            n_atoms,
            c_z,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def time_diffusion_conditioning_backward(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark backward pass of DiffusionConditioning."""
        # Zero gradients
        for tensor in [self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        s_out, z_out = self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )
        loss = s_out.sum() + z_out.sum()

        # Backward pass
        loss.backward()

        return loss

    def peakmem_diffusion_conditioning_backward(
        self, batch_size, n_atoms, c_z, c_s, c_s_inputs, dtype
    ):
        """Benchmark peak memory usage of DiffusionConditioning backward pass."""
        # Zero gradients
        for tensor in [self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk]:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Forward pass
        s_out, z_out = self.module(
            self.t, self.f_star, self.s_inputs, self.s_trunk, self.z_trunk
        )
        loss = s_out.sum() + z_out.sum()

        # Backward pass
        loss.backward()

        return loss


class TimeDiffusionConditioningComponents:
    """Benchmark individual components of DiffusionConditioning."""

    params = ([32, 64], [128], [256], [128], [torch.float32])
    param_names = ["n_atoms", "c_z", "c_s", "c_s_inputs", "dtype"]

    def setup(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        device = torch.device("cpu")
        batch_size = 2

        # Create module
        self.module = (
            DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs)
            .to(device)
            .to(dtype)
        )

        # Generate test data
        self.t = torch.randn(batch_size, 1, dtype=dtype, device=device).abs() + 0.1
        self.f_star = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.s_inputs = torch.randn(
            batch_size, n_atoms, c_s_inputs, dtype=dtype, device=device
        )
        self.s_trunk = torch.randn(batch_size, n_atoms, c_s, dtype=dtype, device=device)
        self.z_trunk = torch.randn(
            batch_size, n_atoms, n_atoms, c_z, dtype=dtype, device=device
        )

    def time_relative_position_encoding(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        """Benchmark RelativePositionEncoding component."""
        return self.module.relative_pos_enc(self.f_star)

    def time_fourier_embedding(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        """Benchmark FourierEmbedding component."""
        t_scaled = self.t / self.module.sigma_data
        t_log = 0.25 * torch.log(torch.clamp(t_scaled, min=1e-8))
        return self.module.fourier_embedding(t_log)

    def time_transition_blocks(self, n_atoms, c_z, c_s, c_s_inputs, dtype):
        """Benchmark Transition blocks."""
        # Test one transition block
        dummy_input = torch.randn(
            2, n_atoms, c_z, device=self.f_star.device, dtype=self.f_star.dtype
        )
        return self.module.transition_z_1(dummy_input) + self.module.transition_z_2(
            dummy_input
        )

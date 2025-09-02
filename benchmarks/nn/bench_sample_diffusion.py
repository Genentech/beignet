import torch

from beignet.nn import SampleDiffusion


class TimeSampleDiffusion:
    """Benchmark SampleDiffusion module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = SampleDiffusion(
            s_trans=1.0,
            gamma_0=1.0,
            gamma_min=0.0001,
            noise_scale=1.0,
        )

        n_tokens = 16  # Small for benchmark
        n_atoms = 32

        # Create feature dictionary
        self.f_star = {
            "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
            "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "asym_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "sym_id": torch.randint(0, 5, (batch_size, n_tokens)),
        }

        self.s_inputs = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.s_trunk = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_trunk = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)

        # Short noise schedule for benchmarking
        self.noise_schedule = torch.linspace(1.0, 0.01, 5, dtype=dtype)

    def time_sample_diffusion(self, batch_size, dtype):
        """Benchmark SampleDiffusion forward pass."""
        return self.module(
            self.f_star, self.s_inputs, self.s_trunk, self.z_trunk, self.noise_schedule
        )


class PeakMemorySampleDiffusion:
    """Benchmark memory usage of SampleDiffusion module."""

    params = ([1], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = SampleDiffusion(
            s_trans=1.0,
            gamma_0=1.0,
            gamma_min=0.0001,
            noise_scale=1.0,
        )

        n_tokens = 8  # Very small for memory test
        n_atoms = 16

        # Create feature dictionary
        self.f_star = {
            "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
            "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "asym_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "sym_id": torch.randint(0, 5, (batch_size, n_tokens)),
        }

        self.s_inputs = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.s_trunk = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_trunk = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)

        # Very short noise schedule for memory test
        self.noise_schedule = torch.linspace(1.0, 0.01, 3, dtype=dtype)

    def peakmem_sample_diffusion(self, batch_size, dtype):
        """Benchmark memory usage of SampleDiffusion forward pass."""
        return self.module(
            self.f_star, self.s_inputs, self.s_trunk, self.z_trunk, self.noise_schedule
        )

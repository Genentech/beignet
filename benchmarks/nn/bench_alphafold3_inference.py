import torch

from beignet.nn import AlphaFold3


class TimeAlphaFold3:
    """Benchmark AlphaFold3 module."""

    params = ([1], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3(
            n_cycle=1,  # Small for benchmark
            c_s=32,
            c_z=16,
            c_m=8,
            n_blocks_pairformer=2,
            n_head=4,
        )

        n_tokens = 16  # Small for benchmark

        # Create minimal required features
        self.f_star = {
            "asym_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "sym_id": torch.randint(0, 5, (batch_size, n_tokens)),
            "token_bonds": torch.randn(batch_size, n_tokens, n_tokens, 32, dtype=dtype),
            "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
            "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
        }

    def time_alphafold3_inference(self, batch_size, dtype):
        """Benchmark AlphaFold3 forward pass."""
        return self.module(self.f_star)


class PeakMemoryAlphaFold3:
    """Benchmark memory usage of AlphaFold3 module."""

    params = ([1], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AlphaFold3(
            n_cycle=1,
            c_s=16,  # Very small for memory test
            c_z=8,
            c_m=4,
            n_blocks_pairformer=1,
            n_head=2,
        )

        n_tokens = 8  # Very small for memory test

        # Create minimal required features
        self.f_star = {
            "asym_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "sym_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "token_bonds": torch.randn(batch_size, n_tokens, n_tokens, 32, dtype=dtype),
            "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
            "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
        }

    def peakmem_alphafold3_inference(self, batch_size, dtype):
        """Benchmark memory usage of AlphaFold3 forward pass."""
        return self.module(self.f_star)

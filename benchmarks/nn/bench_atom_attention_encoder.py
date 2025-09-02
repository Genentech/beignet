import torch

from beignet.nn import AtomAttentionEncoder


class TimeAtomAttentionEncoder:
    """Benchmark AtomAttentionEncoder module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AtomAttentionEncoder(
            c_s=32,
            c_z=16,
            c_atom=8,
            c_atompair=4,
            n_head=4,
        )

        n_tokens = 32
        n_atoms = 64

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

        self.r_t = torch.randn(batch_size, n_atoms, 3, dtype=dtype)
        self.s_trunk = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)

    def time_atom_attention_encoder(self, batch_size, dtype):
        """Benchmark AtomAttentionEncoder forward pass."""
        return self.module(self.f_star, self.r_t, self.s_trunk, self.z_ij)


class PeakMemoryAtomAttentionEncoder:
    """Benchmark memory usage of AtomAttentionEncoder module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = AtomAttentionEncoder(
            c_s=32,
            c_z=16,
            c_atom=8,
            c_atompair=4,
            n_head=4,
        )

        n_tokens = 16
        n_atoms = 32

        # Create feature dictionary (smaller for memory test)
        self.f_star = {
            "atom_coordinates": torch.randn(batch_size, n_tokens, 3, dtype=dtype),
            "atom_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "asym_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "entity_id": torch.randint(0, 2, (batch_size, n_tokens)),
            "sym_id": torch.randint(0, 5, (batch_size, n_tokens)),
        }

        self.r_t = torch.randn(batch_size, n_atoms, 3, dtype=dtype)
        self.s_trunk = torch.randn(batch_size, n_tokens, 32, dtype=dtype)
        self.z_ij = torch.randn(batch_size, n_tokens, n_tokens, 16, dtype=dtype)

    def peakmem_atom_attention_encoder(self, batch_size, dtype):
        """Benchmark memory usage of AtomAttentionEncoder forward pass."""
        return self.module(self.f_star, self.r_t, self.s_trunk, self.z_ij)

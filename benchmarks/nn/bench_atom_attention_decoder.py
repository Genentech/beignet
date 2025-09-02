import torch

from beignet.nn import _AtomAttentionDecoder


class TimeAtomAttentionDecoder:
    """Benchmark AtomAttentionDecoder module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = _AtomAttentionDecoder(
            c_token=64,
            c_atom=32,
            n_queries=8,
            n_keys=16,
        )

        n_tokens = 32
        n_atoms = 64

        self.a = torch.randn(batch_size, n_tokens, 64, dtype=dtype)
        self.q_skip = torch.randn(batch_size, n_atoms, 32, dtype=dtype)
        self.c_skip = torch.randn(batch_size, n_atoms, 32, dtype=dtype)
        self.p_skip = torch.randn(batch_size, n_atoms, n_atoms, 16, dtype=dtype)

    def time_atom_attention_decoder(self, batch_size, dtype):
        """Benchmark AtomAttentionDecoder forward pass."""
        return self.module(self.a, self.q_skip, self.c_skip, self.p_skip)


class PeakMemoryAtomAttentionDecoder:
    """Benchmark memory usage of AtomAttentionDecoder module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = _AtomAttentionDecoder(
            c_token=64,
            c_atom=32,
            n_queries=8,
            n_keys=16,
        )

        n_tokens = 16
        n_atoms = 32

        # Smaller for memory test
        self.a = torch.randn(batch_size, n_tokens, 64, dtype=dtype)
        self.q_skip = torch.randn(batch_size, n_atoms, 32, dtype=dtype)
        self.c_skip = torch.randn(batch_size, n_atoms, 32, dtype=dtype)
        self.p_skip = torch.randn(batch_size, n_atoms, n_atoms, 16, dtype=dtype)

    def peakmem_atom_attention_decoder(self, batch_size, dtype):
        """Benchmark memory usage of AtomAttentionDecoder forward pass."""
        return self.module(self.a, self.q_skip, self.c_skip, self.p_skip)

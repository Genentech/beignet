import torch

from beignet.nn import AtomTransformer


class TimeAtomTransformer:
    """Benchmark AtomTransformer module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = batch_size
        self.n_atoms = 64
        self.c_q = 16
        self.c_kv = 16
        self.c_pair = 8

        self.module = AtomTransformer(
            n_block=2,
            n_head=2,
            n_queries=8,
            n_keys=16,
            subset_centres=[8.0, 24.0],
            c_q=self.c_q,
            c_kv=self.c_kv,
            c_pair=self.c_pair,
        )

        self.q = torch.randn(batch_size, self.n_atoms, self.c_q, dtype=dtype)
        self.c = torch.randn(batch_size, self.n_atoms, self.c_kv, dtype=dtype)
        self.p = torch.randn(
            batch_size, self.n_atoms, self.n_atoms, self.c_pair, dtype=dtype
        )

    def time_atom_transformer(self, batch_size, dtype):
        """Benchmark AtomTransformer forward pass."""
        return self.module(self.q, self.c, self.p)


class PeakMemoryAtomTransformer:
    """Benchmark memory usage of AtomTransformer module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = batch_size
        self.n_atoms = 32  # Smaller for memory test
        self.c_q = 16
        self.c_kv = 16
        self.c_pair = 8

        self.module = AtomTransformer(
            n_block=1,
            n_head=2,
            n_queries=4,
            n_keys=8,
            subset_centres=[4.0],
            c_q=self.c_q,
            c_kv=self.c_kv,
            c_pair=self.c_pair,
        )

        self.q = torch.randn(batch_size, self.n_atoms, self.c_q, dtype=dtype)
        self.c = torch.randn(batch_size, self.n_atoms, self.c_kv, dtype=dtype)
        self.p = torch.randn(
            batch_size, self.n_atoms, self.n_atoms, self.c_pair, dtype=dtype
        )

    def peakmem_atom_transformer(self, batch_size, dtype):
        """Benchmark memory usage of AtomTransformer forward pass."""
        return self.module(self.q, self.c, self.p)

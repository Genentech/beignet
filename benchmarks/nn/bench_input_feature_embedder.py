import torch

from beignet.nn import _InputFeatureEmbedder


class TimeInputFeatureEmbedder:
    """Benchmark InputFeatureEmbedder module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = _InputFeatureEmbedder(
            c_s=32,
            c_z=16,
            c_atom=8,
            c_atompair=4,
            n_head=4,
        )

        n_tokens = 32

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

    def time_input_feature_embedder(self, batch_size, dtype):
        """Benchmark InputFeatureEmbedder forward pass."""
        return self.module(self.f_star)


class PeakMemoryInputFeatureEmbedder:
    """Benchmark memory usage of InputFeatureEmbedder module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = _InputFeatureEmbedder(
            c_s=32,
            c_z=16,
            c_atom=8,
            c_atompair=4,
            n_head=4,
        )

        n_tokens = 16  # Smaller for memory test

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

    def peakmem_input_feature_embedder(self, batch_size, dtype):
        """Benchmark memory usage of InputFeatureEmbedder forward pass."""
        return self.module(self.f_star)

import torch

from beignet.nn import RelativePositionEncoding


class TimeRelativePositionEncoding:
    """Benchmark RelativePositionEncoding module."""

    params = ([1, 4], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)

        n_tokens = 64

        # Create feature dictionary
        self.f_star = {
            "asym_id": torch.randint(0, 5, (batch_size, n_tokens)),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "entity_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "sym_id": torch.randint(0, 10, (batch_size, n_tokens)),
        }

    def time_relative_position_encoding(self, batch_size, dtype):
        """Benchmark RelativePositionEncoding forward pass."""
        return self.module(self.f_star)


class PeakMemoryRelativePositionEncoding:
    """Benchmark memory usage of RelativePositionEncoding module."""

    params = ([1, 2], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = RelativePositionEncoding(
            r_max=16,  # Smaller for memory test
            s_max=2,
            c_z=64,  # Smaller for memory test
        )

        n_tokens = 32  # Smaller for memory test

        # Create feature dictionary
        self.f_star = {
            "asym_id": torch.randint(0, 5, (batch_size, n_tokens)),
            "residue_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "entity_id": torch.randint(0, 3, (batch_size, n_tokens)),
            "token_index": torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
            "sym_id": torch.randint(0, 10, (batch_size, n_tokens)),
        }

    def peakmem_relative_position_encoding(self, batch_size, dtype):
        """Benchmark memory usage of RelativePositionEncoding forward pass."""
        return self.module(self.f_star)

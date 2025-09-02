import torch

from beignet.nn import CentreRandomAugmentation


class TimeCentreRandomAugmentation:
    """Benchmark CentreRandomAugmentation module."""

    params = ([1, 10, 100], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = CentreRandomAugmentation(s_trans=1.0)
        self.x_t = torch.randn(batch_size, 128, 3, dtype=dtype)

    def time_centre_random_augmentation(self, batch_size, dtype):
        """Benchmark CentreRandomAugmentation forward pass."""
        return self.module(self.x_t)


class PeakMemoryCentreRandomAugmentation:
    """Benchmark memory usage of CentreRandomAugmentation module."""

    params = ([1, 10], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        """Setup test data."""
        torch.manual_seed(42)
        self.module = CentreRandomAugmentation(s_trans=1.0)
        self.x_t = torch.randn(
            batch_size, 64, 3, dtype=dtype
        )  # Smaller for memory test

    def peakmem_centre_random_augmentation(self, batch_size, dtype):
        """Benchmark memory usage of CentreRandomAugmentation forward pass."""
        return self.module(self.x_t)

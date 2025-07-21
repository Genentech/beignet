import torch

import beignet


class TimeRelativePermittivity:
    params = (
        [1, 10, 100],
        [8, 16, 32],
        [1, 3, 8],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "spatial_dim", "channels", "dtype"]

    def setup(self, batch_size, spatial_dim, channels, dtype):
        torch.manual_seed(42)
        # Create 2D spatial inputs
        self.input = torch.randn(
            batch_size, channels, spatial_dim, spatial_dim, dtype=dtype
        )
        self.charges = torch.randn(batch_size, 1, 1, dtype=dtype) * 10.0
        self.temperature = torch.full((batch_size,), 300.0, dtype=dtype)

        # Compile the function
        self.compiled_fn = torch.compile(beignet.relative_permittivity, fullgraph=True)

        # Warm up
        _ = self.compiled_fn(self.input, self.charges, self.temperature)

    def time_relative_permittivity(self, batch_size, spatial_dim, channels, dtype):
        beignet.relative_permittivity(self.input, self.charges, self.temperature)

    def time_relative_permittivity_compiled(
        self, batch_size, spatial_dim, channels, dtype
    ):
        self.compiled_fn(self.input, self.charges, self.temperature)


class PeakMemoryRelativePermittivity:
    params = (
        [1, 10, 100],
        [8, 16, 32],
        [1, 3, 8],
        [torch.float32, torch.float64],
    )
    param_names = ["batch_size", "spatial_dim", "channels", "dtype"]

    def setup(self, batch_size, spatial_dim, channels, dtype):
        torch.manual_seed(42)
        self.input = torch.randn(
            batch_size, channels, spatial_dim, spatial_dim, dtype=dtype
        )
        self.charges = torch.randn(batch_size, 1, 1, dtype=dtype) * 10.0
        self.temperature = torch.full((batch_size,), 300.0, dtype=dtype)

    def peakmem_relative_permittivity(self, batch_size, spatial_dim, channels, dtype):
        beignet.relative_permittivity(self.input, self.charges, self.temperature)

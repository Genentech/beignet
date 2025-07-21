import torch

import beignet


class TimeIsothermalCompressibility:
    params = ([1, 10, 100], [100, 1000, 10000], [torch.float32, torch.float64])
    param_names = ["batch_size", "num_timesteps", "dtype"]

    def setup(self, batch_size, num_timesteps, dtype):
        torch.manual_seed(42)
        self.volumes = torch.randn(batch_size, num_timesteps, dtype=dtype) * 100 + 5000
        self.temperature = torch.full((batch_size,), 300.0, dtype=dtype)

        # Compile the function for benchmarking
        self.compiled_fn = torch.compile(
            beignet.isothermal_compressibility, fullgraph=True
        )

        # Warm up the compiled function
        _ = self.compiled_fn(self.volumes, self.temperature)

    def time_isothermal_compressibility(self, batch_size, num_timesteps, dtype):
        beignet.isothermal_compressibility(self.volumes, self.temperature)

    def time_isothermal_compressibility_compiled(
        self, batch_size, num_timesteps, dtype
    ):
        self.compiled_fn(self.volumes, self.temperature)


class PeakMemoryIsothermalCompressibility:
    params = ([1, 10, 100], [100, 1000, 10000], [torch.float32, torch.float64])
    param_names = ["batch_size", "num_timesteps", "dtype"]

    def setup(self, batch_size, num_timesteps, dtype):
        torch.manual_seed(42)
        self.volumes = torch.randn(batch_size, num_timesteps, dtype=dtype) * 100 + 5000
        self.temperature = torch.full((batch_size,), 300.0, dtype=dtype)

    def peakmem_isothermal_compressibility(self, batch_size, num_timesteps, dtype):
        beignet.isothermal_compressibility(self.volumes, self.temperature)

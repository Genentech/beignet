import os

import torch

import beignet


class TimeThermalExpansion:
    params = ([1, 10, 100], [100, 1000, 10000], [torch.float32, torch.float64])
    param_names = ["batch_size", "num_timesteps", "dtype"]

    def setup(self, batch_size, num_timesteps, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))
        self.volumes = torch.randn(batch_size, num_timesteps, dtype=dtype) * 10 + 5000
        self.enthalpies = (
            torch.randn(batch_size, num_timesteps, dtype=dtype) * 100 - 5000
        )
        self.temperatures = torch.full((batch_size,), 300.0, dtype=dtype)

    def time_thermal_expansion(self, batch_size, num_timesteps, dtype):
        beignet.thermal_expansion(self.volumes, self.temperatures, self.enthalpies)


class PeakMemoryThermalExpansion:
    params = ([1, 10, 100], [100, 1000, 10000], [torch.float32, torch.float64])
    param_names = ["batch_size", "num_timesteps", "dtype"]

    def setup(self, batch_size, num_timesteps, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))
        self.volumes = torch.randn(batch_size, num_timesteps, dtype=dtype) * 10 + 5000
        self.enthalpies = (
            torch.randn(batch_size, num_timesteps, dtype=dtype) * 100 - 5000
        )
        self.temperatures = torch.full((batch_size,), 300.0, dtype=dtype)

    def peakmem_thermal_expansion(self, batch_size, num_timesteps, dtype):
        beignet.thermal_expansion(self.volumes, self.temperatures, self.enthalpies)


class TimeThermalExpansionCompiled:
    params = ([1, 10], [1000, 10000], [torch.float32, torch.float64])
    param_names = ["batch_size", "num_timesteps", "dtype"]

    def setup(self, batch_size, num_timesteps, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))
        self.volumes = torch.randn(batch_size, num_timesteps, dtype=dtype) * 10 + 5000
        self.enthalpies = (
            torch.randn(batch_size, num_timesteps, dtype=dtype) * 100 - 5000
        )
        self.temperatures = torch.full((batch_size,), 300.0, dtype=dtype)
        self.compiled_fn = torch.compile(beignet.thermal_expansion, fullgraph=True)

    def time_thermal_expansion_compiled(self, batch_size, num_timesteps, dtype):
        self.compiled_fn(self.volumes, self.temperatures, self.enthalpies)

import os

import torch

import beignet


class TimePhiCoefficient:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.chi_square_values = (
            torch.tensor([1.0, 4.0, 9.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_phi_coefficient = torch.compile(
            beignet.phi_coefficient, fullgraph=True
        )

    def time_phi_coefficient(self, batch_size, dtype):
        return self.compiled_phi_coefficient(self.chi_square_values, self.sample_sizes)


class PeakMemoryPhiCoefficient:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.chi_square_values = (
            torch.tensor([1.0, 4.0, 9.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_phi_coefficient(self, batch_size, dtype):
        return beignet.phi_coefficient(self.chi_square_values, self.sample_sizes)

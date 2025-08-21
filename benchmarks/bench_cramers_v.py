import os

import torch

import beignet


class TimeCramersV:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.chi_square_values = (
            torch.tensor([1.0, 5.5, 12.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.min_dims = (
            torch.tensor([1, 2, 3], dtype=dtype).repeat(batch_size, 1).flatten()
        )

        # Compile for optimal performance
        self.compiled_cramers_v = torch.compile(beignet.cramers_v, fullgraph=True)

    def time_cramers_v(self, batch_size, dtype):
        return self.compiled_cramers_v(
            self.chi_square_values, self.sample_sizes, self.min_dims
        )


class PeakMemoryCramersV:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.chi_square_values = (
            torch.tensor([1.0, 5.5, 12.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.min_dims = (
            torch.tensor([1, 2, 3], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def peakmem_cramers_v(self, batch_size, dtype):
        return beignet.cramers_v(
            self.chi_square_values, self.sample_sizes, self.min_dims
        )

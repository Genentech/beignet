import os

import torch

import beignet


class CohensF:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        # Create group means for 3 groups
        self.group_means = (
            torch.tensor(
                [[10.0, 12.0, 14.0], [5.0, 7.0, 9.0], [20.0, 22.0, 24.0]], dtype=dtype
            )
            .repeat(batch_size, 1, 1)
            .view(-1, 3)
        )

        self.pooled_stds = (
            torch.tensor([2.0, 1.5, 3.0], dtype=dtype).repeat(batch_size).flatten()
        )

        # Compile for optimal performance
        self.compiled_cohens_f = torch.compile(beignet.cohens_f, fullgraph=True)

    def time_cohens_f(self, batch_size, dtype):
        return self.compiled_cohens_f(self.group_means, self.pooled_stds)


class PeakMemoryCohensF:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        self.group_means = (
            torch.tensor(
                [[10.0, 12.0, 14.0], [5.0, 7.0, 9.0], [20.0, 22.0, 24.0]], dtype=dtype
            )
            .repeat(batch_size, 1, 1)
            .view(-1, 3)
        )

        self.pooled_stds = (
            torch.tensor([2.0, 1.5, 3.0], dtype=dtype).repeat(batch_size).flatten()
        )

    def peakmem_cohens_f(self, batch_size, dtype):
        return beignet.cohens_f(self.group_means, self.pooled_stds)

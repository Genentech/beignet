import os

import torch

import beignet


class TimeCohensD:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        n1, n2 = 50, 50
        self.group1 = torch.randn(batch_size, n1, dtype=dtype)
        self.group2 = torch.randn(batch_size, n2, dtype=dtype)

        # Compile for optimal performance
        self.compiled_cohens_d = torch.compile(beignet.cohens_d, fullgraph=True)

    def time_cohens_d(self, batch_size, dtype):
        return self.compiled_cohens_d(self.group1, self.group2, pooled=True)

    def time_cohens_d_nonpooled(self, batch_size, dtype):
        return self.compiled_cohens_d(self.group1, self.group2, pooled=False)


class PeakMemoryCohensD:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        seed = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
        torch.manual_seed(seed)

        n1, n2 = 50, 50
        self.group1 = torch.randn(batch_size, n1, dtype=dtype)
        self.group2 = torch.randn(batch_size, n2, dtype=dtype)

    def peakmem_cohens_d(self, batch_size, dtype):
        return beignet.cohens_d(self.group1, self.group2, pooled=True)

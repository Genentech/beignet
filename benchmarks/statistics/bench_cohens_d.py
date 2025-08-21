import torch

import beignet

from ._set_seed import set_seed


class BenchCohensD:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.cohens_d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        n1, n2 = 50, 50
        self.group1 = torch.randn(batch_size, n1, dtype=dtype)
        self.group2 = torch.randn(batch_size, n2, dtype=dtype)

    def time_cohens_d(self, batch_size, dtype):
        return self.func(self.group1, self.group2, pooled=True)

    def time_cohens_d_nonpooled(self, batch_size, dtype):
        return self.func(self.group1, self.group2, pooled=False)

    def peakmem_cohens_d(self, batch_size, dtype):
        return beignet.statistics.cohens_d(self.group1, self.group2, pooled=True)

import torch

import beignet

from ._set_seed import set_seed


class BenchHedgesG:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.hedges_g,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        n1, n2 = 50, 50
        self.group1 = torch.randn(batch_size, n1, dtype=dtype)
        self.group2 = torch.randn(batch_size, n2, dtype=dtype)

    def time_hedges_g(self, batch_size, dtype):
        return self.func(self.group1, self.group2)

    def peakmem_hedges_g(self, batch_size, dtype):
        return beignet.statistics.hedges_g(self.group1, self.group2)

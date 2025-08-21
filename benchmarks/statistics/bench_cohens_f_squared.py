import torch

import beignet

from ._set_seed import set_seed


class BenchCohensFSquared:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.cohens_f_squared,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

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

    def time_cohens_f_squared(self, batch_size, dtype):
        return self.func(self.group_means, self.pooled_stds)

    def peakmem_cohens_f_squared(self, batch_size, dtype):
        return beignet.statistics.cohens_f_squared(self.group_means, self.pooled_stds)

import torch

import beignet

from ._set_seed import set_seed


class BenchProportionTwoSampleSampleSize:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.proportion_two_sample_sample_size,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.p1_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p2_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_proportion_two_sample_sample_size_two_sided(self, batch_size, dtype):
        return self.func(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_greater(self, batch_size, dtype):
        return self.func(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="greater",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_less(self, batch_size, dtype):
        return self.func(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="less",
            ratio=1.0,
        )

    def time_proportion_two_sample_sample_size_unequal(self, batch_size, dtype):
        return self.func(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=2.0,
        )

    def peakmem_proportion_two_sample_sample_size(self, batch_size, dtype):
        return beignet.statistics.proportion_two_sample_sample_size(
            self.p1_values,
            self.p2_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
            ratio=1.0,
        )

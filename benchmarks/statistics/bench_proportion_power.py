import torch

import beignet

from ._set_seed import set_seed


class BenchProportionPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.proportion_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.p0_values = (
            torch.tensor([0.4, 0.5, 0.6], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.p1_values = (
            torch.tensor([0.5, 0.6, 0.7], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_proportion_power_two_sided(self, batch_size, dtype):
        return self.func(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

    def time_proportion_power_greater(self, batch_size, dtype):
        return self.func(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="greater",
        )

    def time_proportion_power_less(self, batch_size, dtype):
        return self.func(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="less",
        )

    def peakmem_proportion_power(self, batch_size, dtype):
        return beignet.statistics.proportion_power(
            self.p0_values,
            self.p1_values,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

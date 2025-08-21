import torch

import beignet

from ._set_seed import set_seed


class BenchCorrelationPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.correlation_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.r_values = (
            torch.tensor([0.1, 0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_correlation_power_two_sided(self, batch_size, dtype):
        return self.func(
            self.r_values, self.sample_sizes, alpha=0.05, alternative="two-sided"
        )

    def time_correlation_power_greater(self, batch_size, dtype):
        return self.func(
            self.r_values, self.sample_sizes, alpha=0.05, alternative="greater"
        )

    def time_correlation_power_less(self, batch_size, dtype):
        return self.func(
            self.r_values, self.sample_sizes, alpha=0.05, alternative="less"
        )

    def peakmem_correlation_power(self, batch_size, dtype):
        return beignet.statistics.correlation_power(
            self.r_values, self.sample_sizes, alpha=0.05, alternative="two-sided"
        )

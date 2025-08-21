import torch

import beignet

from ._set_seed import set_seed


class BenchTTestSampleSize:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.statistics.t_test_sample_size,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_t_test_sample_size_two_sided(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )

    def time_t_test_sample_size_greater(self, batch_size, dtype):
        return self.func(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="greater"
        )

    def time_t_test_sample_size_less(self, batch_size, dtype):
        return self.func(self.effect_sizes, power=0.8, alpha=0.05, alternative="less")

    def peakmem_t_test_sample_size(self, batch_size, dtype):
        return beignet.statistics.t_test_sample_size(
            self.effect_sizes, power=0.8, alpha=0.05, alternative="two-sided"
        )
